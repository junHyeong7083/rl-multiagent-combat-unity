import argparse
import socket
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from multiagent_env_5v5 import CombatSelfPlay5v5Env
from ac_model import TeamPolicy, UnitActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(width, height, n_per_team, max_steps, hp, reload_steps, seed):
    return CombatSelfPlay5v5Env(
        width=width, height=height, n_per_team=n_per_team,
        max_steps=max_steps, hp=hp, reload_steps=reload_steps, seed=seed
    )

def obs_dim_for_env(env):
    return env._obs_team(env.A, env.B).shape[0]

class UdpStreamer:
    def __init__(self, port: int | None):
        self.port = port
        self.sock = None
        self.addr = ("127.0.0.1", port) if port else None
        if port:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_env(self, env, info):
        if not self.sock: return
        payload = {
            "t": env.t,
            "width": env.width,
            "height": env.height,
            "baseA": info.get("base_A"),
            "baseB": info.get("base_B"),
            "A": env.A.tolist(),
            "B": env.B.tolist(),
            "shots": info.get("shots", []),
            "outcome": info.get("outcome"),
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)

def rollout(env, teamA, unitA, teamB, unitB, num_tactics, T, streamer: UdpStreamer, gamma=0.99):
    obsA, obsB = env.reset()
    obsA = torch.from_numpy(obsA).float().to(device)
    obsB = torch.from_numpy(obsB).float().to(device)

    stor = dict(
        obsA=[], obsB=[], zA=[], zB=[], aA=[], aB=[],
        logpA=[], logpB=[], vA_team=[], vB_team=[], vA_unit=[], vB_unit=[],
        rA=[], rB=[], done=[], outcome=[]
    )

    ep_ret_A = 0.0
    ep_ret_B = 0.0
    ep_cnt = 0
    outcome_hist = {"A_capture":0, "B_capture":0, "A_wipe":0, "B_wipe":0, "timeout":0, None:0}

    for t in range(T):
        # team tactics
        logitsA, vA_t = teamA(obsA.unsqueeze(0))
        logitsB, vB_t = teamB(obsB.unsqueeze(0))
        distA = Categorical(logits=logitsA.squeeze(0))
        distB = Categorical(logits=logitsB.squeeze(0))
        zA = distA.sample()
        zB = distB.sample()
        zA_oh = F.one_hot(zA, num_classes=num_tactics).float().to(device)
        zB_oh = F.one_hot(zB, num_classes=num_tactics).float().to(device)

        # unit actions (shared policy, n independent samples)
        logits_uA, vA_u = unitA(obsA.unsqueeze(0), zA_oh.unsqueeze(0))
        logits_uB, vB_u = unitB(obsB.unsqueeze(0), zB_oh.unsqueeze(0))
        piA = Categorical(logits=logits_uA.squeeze(0))
        piB = Categorical(logits=logits_uB.squeeze(0))

        n = env.n
        aA = piA.sample((n,))
        aB = piB.sample((n,))

        next_obsA, next_obsB, rA, rB, done, info = env.step(aA.cpu().numpy(), aB.cpu().numpy())

        # stream to Unity (optional)
        if streamer: streamer.send_env(env, info)

        stor['obsA'].append(obsA); stor['obsB'].append(obsB)
        stor['zA'].append(zA); stor['zB'].append(zB)
        stor['aA'].append(aA); stor['aB'].append(aB)
        stor['logpA'].append(distA.log_prob(zA) + piA.log_prob(aA).sum())
        stor['logpB'].append(distB.log_prob(zB) + piB.log_prob(aB).sum())
        stor['vA_team'].append(vA_t.squeeze(0)); stor['vB_team'].append(vB_t.squeeze(0))
        stor['vA_unit'].append(vA_u.squeeze(0)); stor['vB_unit'].append(vB_u.squeeze(0))
        stor['rA'].append(torch.tensor(rA, dtype=torch.float32, device=device))
        stor['rB'].append(torch.tensor(rB, dtype=torch.float32, device=device))
        stor['done'].append(torch.tensor(done, dtype=torch.float32, device=device))
        stor['outcome'].append(info.get("outcome"))

        ep_ret_A += rA
        ep_ret_B += rB

        obsA = torch.from_numpy(next_obsA).float().to(device)
        obsB = torch.from_numpy(next_obsB).float().to(device)

        if done:
            ep_cnt += 1
            outcome_hist[info.get("outcome")] = outcome_hist.get(info.get("outcome"), 0) + 1
            ep_ret_A = 0.0
            ep_ret_B = 0.0
            obsA, obsB = env.reset()
            obsA = torch.from_numpy(obsA).float().to(device)
            obsB = torch.from_numpy(obsB).float().to(device)

    # returns/advantages (간단 버전)
    R_A = 0; R_B = 0
    advA = []; advB = []
    retA = []; retB = []
    for t in reversed(range(T)):
        R_A = stor['rA'][t] + gamma * R_A * (1 - stor['done'][t])
        R_B = stor['rB'][t] + gamma * R_B * (1 - stor['done'][t])
        retA.append(R_A); retB.append(R_B)

        vA = stor['vA_team'][t] + stor['vA_unit'][t]
        vB = stor['vB_team'][t] + stor['vB_unit'][t]
        # one-step TD advantage (간결화)
        deltaA = stor['rA'][t] + gamma * (0 if stor['done'][t] else vA.detach()) - vA.detach()
        deltaB = stor['rB'][t] + gamma * (0 if stor['done'][t] else vB.detach()) - vB.detach()
        advA.insert(0, deltaA)
        advB.insert(0, deltaB)

    for k in ['obsA','obsB','zA','zB','aA','aB','logpA','logpB','vA_team','vB_team','vA_unit','vB_unit']:
        stor[k] = torch.stack(stor[k])
    stor['retA'] = torch.stack(retA[::-1]); stor['retB'] = torch.stack(retB[::-1])
    stor['advA'] = torch.stack(advA); stor['advB'] = torch.stack(advB)
    stor['outcome_hist'] = outcome_hist
    stor['episodes'] = ep_cnt
    return stor

def train(args):
    env = make_env(args.width, args.height, args.n_per_team, args.max_steps, args.hp, args.reload_steps, args.seed)
    obs_dim = obs_dim_for_env(env)

    teamA = TeamPolicy(obs_dim, args.num_tactics).to(device)
    teamB = TeamPolicy(obs_dim, args.num_tactics).to(device)
    unitA = UnitActorCritic(obs_dim, args.num_tactics).to(device)
    unitB = UnitActorCritic(obs_dim, args.num_tactics).to(device)

    opt = optim.Adam(list(teamA.parameters())+list(teamB.parameters())+
                     list(unitA.parameters())+list(unitB.parameters()), lr=args.lr)

    streamer = UdpStreamer(args.stream_udp) if args.stream_udp else None

    ema_retA = None
    ema_retB = None
    ema_alpha = 0.1

    for step in range(1, args.steps+1):
        stor = rollout(env, teamA, unitA, teamB, unitB,
                       num_tactics=args.num_tactics, T=args.rollout_len,
                       streamer=streamer)

        # policy/value losses
        logpA = stor['logpA']; logpB = stor['logpB']
        advA = (stor['advA'] - stor['advA'].mean()) / (stor['advA'].std()+1e-5)
        advB = (stor['advB'] - stor['advB'].mean()) / (stor['advB'].std()+1e-5)

        vA = stor['vA_team'] + stor['vA_unit']
        vB = stor['vB_team'] + stor['vB_unit']

        val_loss = F.mse_loss(vA, stor['retA']) + F.mse_loss(vB, stor['retB'])
        pi_loss = -(logpA*advA).mean() - (logpB*advB).mean()
        loss = pi_loss + 0.5*val_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(teamA.parameters())+list(teamB.parameters())+
                                       list(unitA.parameters())+list(unitB.parameters()), max_norm=1.0)
        opt.step()

        # 간단한 성능 지표: episode 수/결말 통계
        ep = max(stor['episodes'], 1)
        oh = stor['outcome_hist']
        # outcome 비율
        captA = oh.get("A_capture", 0); captB = oh.get("B_capture", 0)
        wipeA = oh.get("A_wipe", 0); wipeB = oh.get("B_wipe", 0)
        tout  = oh.get("timeout", 0)

        # EMA (여기선 outcome 카운트로 대체)
        if ema_retA is None:
            ema_retA = captA - captB + wipeA - wipeB
            ema_retB = captB - captA + wipeB - wipeA
        else:
            ema_retA = (1-ema_alpha)*ema_retA + ema_alpha*(captA - captB + wipeA - wipeB)
            ema_retB = (1-ema_alpha)*ema_retB + ema_alpha*(captB - captA + wipeB - wipeA)

        if step % args.log_every == 0:
            print(
                f"step {step} | loss={loss.item():.3f} | "
                f"episodes={ep} | "
                f"A_cap={captA} B_cap={captB} A_wipe={wipeA} B_wipe={wipeB} TO={tout} | "
                f"EMA(A)={ema_retA:.2f} EMA(B)={ema_retB:.2f}"
            )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--rollout_len", type=int, default=64)
    ap.add_argument("--num_tactics", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    # env
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--n_per_team", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=240)
    ap.add_argument("--hp", type=int, default=3)
    ap.add_argument("--reload_steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # streaming
    ap.add_argument("--stream_udp", type=int, default=0, help="UDP port for Unity viewer (0=off).")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    train(args)
