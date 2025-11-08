import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from multiagent_env_5v5 import CombatSelfPlay5v5Env
from ac_model import TeamTacticActorCritic, UnitActorCriticMultiHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    def __init__(self): self.clear()
    def clear(self):
        # 팀 A
        self.obs_team_A=[]; self.zA=[]; self.unit_obs_A=[]; self.unit_id_A=[]
        self.act_A=[]; self.logp_A=[]; self.v_A=[]; self.rew_A=[]
        # 팀 B
        self.obs_team_B=[]; self.zB=[]; self.unit_obs_B=[]; self.unit_id_B=[]
        self.act_B=[]; self.logp_B=[]; self.v_B=[]; self.rew_B=[]
        self.done=[]
    def to_tensors(self):
        def cat(xs): return torch.cat(xs, dim=0) if xs else None
        out = {
          "obs_team_A": torch.stack(self.obs_team_A,0).to(DEVICE),
          "zA": torch.tensor(self.zA, dtype=torch.long, device=DEVICE),
          "unit_obs_A": cat(self.unit_obs_A).to(DEVICE),
          "unit_id_A": cat(self.unit_id_A).to(DEVICE),
          "act_A": cat(self.act_A).long().to(DEVICE),
          "logp_A": cat(self.logp_A).to(DEVICE),
          "v_A": cat(self.v_A).to(DEVICE),
          "rew_A": torch.tensor(self.rew_A, dtype=torch.float32, device=DEVICE),

          "obs_team_B": torch.stack(self.obs_team_B,0).to(DEVICE),
          "zB": torch.tensor(self.zB, dtype=torch.long, device=DEVICE),
          "unit_obs_B": cat(self.unit_obs_B).to(DEVICE),
          "unit_id_B": cat(self.unit_id_B).to(DEVICE),
          "act_B": cat(self.act_B).long().to(DEVICE),
          "logp_B": cat(self.logp_B).to(DEVICE),
          "v_B": cat(self.v_B).to(DEVICE),
          "rew_B": torch.tensor(self.rew_B, dtype=torch.float32, device=DEVICE),

          "done": torch.tensor(self.done, dtype=torch.float32, device=DEVICE),
        }
        return out


def make_unit_id_onehots(n):
    eye = torch.eye(n, device=DEVICE)
    return [eye[i:i+1,:] for i in range(n)]


def rollout(env, teamA, unitA, teamB, unitB, horizon, num_tactics):
    buf = RolloutBuffer()
    team_obs_A, team_obs_B = env.reset()
    unit_id_onehots = make_unit_id_onehots(env.n)

    for _ in range(horizon):
        tA = torch.tensor(team_obs_A, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        tB = torch.tensor(team_obs_B, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # --- 팀 전술: A/B 서로 다른 네트워크 ---
        logits_zA, v_team_A = teamA(tA)
        logits_zB, v_team_B = teamB(tB)
        zA = Categorical(logits=logits_zA).sample().item()
        zB = Categorical(logits=logits_zB).sample().item()

        zA_oh = torch.nn.functional.one_hot(torch.tensor([zA], device=DEVICE), num_classes=num_tactics).float()
        zB_oh = torch.nn.functional.one_hot(torch.tensor([zB], device=DEVICE), num_classes=num_tactics).float()

        # --- 유닛 정책: 팀마다 다른 멀티헤드 ---
        tA_rep = tA.repeat(env.n,1); zA_rep = zA_oh.repeat(env.n,1)
        tB_rep = tB.repeat(env.n,1); zB_rep = zB_oh.repeat(env.n,1)
        uidA = torch.cat(unit_id_onehots, dim=0)
        uidB = torch.cat(unit_id_onehots, dim=0)

        logits_uA, vA = unitA(tA_rep, zA_rep, uidA)
        logits_uB, vB = unitB(tB_rep, zB_rep, uidB)
        pi_uA = Categorical(logits=logits_uA)
        pi_uB = Categorical(logits=logits_uB)
        aA = pi_uA.sample(); aB = pi_uB.sample()
        logpA = pi_uA.log_prob(aA); logpB = pi_uB.log_prob(aB)

        aA_np = aA.detach().cpu().numpy().astype(np.int32)
        aB_np = aB.detach().cpu().numpy().astype(np.int32)
        nextA, nextB, rA, rB, done, info = env.step(aA_np, aB_np)

        # ---- 버퍼 적재 (A/B 분리) ----
        buf.obs_team_A.append(tA.squeeze(0))
        buf.zA.append(zA)
        buf.unit_obs_A.append(tA_rep.detach())
        buf.unit_id_A.append(uidA.detach())
        buf.act_A.append(aA.detach())
        buf.logp_A.append(logpA.detach())
        buf.v_A.append(vA.detach())
        buf.rew_A.append(rA)

        buf.obs_team_B.append(tB.squeeze(0))
        buf.zB.append(zB)
        buf.unit_obs_B.append(tB_rep.detach())
        buf.unit_id_B.append(uidB.detach())
        buf.act_B.append(aB.detach())
        buf.logp_B.append(logpB.detach())
        buf.v_B.append(vB.detach())
        buf.rew_B.append(rB)

        buf.done.append(1.0 if done else 0.0)

        team_obs_A, team_obs_B = nextA, nextB
        if done:
            team_obs_A, team_obs_B = env.reset()

    return buf


def compute_advantages(roll, gamma=0.99, n_units=5):
    T = roll["rew_A"].shape[0]; dones = roll["done"]
    def returns(rews):
        ret = torch.zeros_like(rews); running = 0.0
        for t in reversed(range(T)):
            running = rews[t] + gamma * running * (1.0 - dones[t]); ret[t] = running
        return ret
    R_A = returns(roll["rew_A"]); R_B = returns(roll["rew_B"])
    vA = roll["v_A"].view(T, n_units); vB = roll["v_B"].view(T, n_units)
    tgtA = R_A.unsqueeze(1).repeat(1, n_units); tgtB = R_B.unsqueeze(1).repeat(1, n_units)
    advA = (tgtA - vA).detach(); advB = (tgtB - vB).detach()
    return advA.reshape(-1), tgtA.reshape(-1), advB.reshape(-1), tgtB.reshape(-1)


def save_ckpt(prefix, team_net, unit_net, meta):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    torch.save({"team": team_net.state_dict(), "unit": unit_net.state_dict(), "meta": meta}, prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--n_per_team", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=180)
    ap.add_argument("--save_dir", type=str, default="ckpt_ab")
    ap.add_argument("--save_every", type=int, default=100)
    args = ap.parse_args()

    env = CombatSelfPlay5v5Env(width=args.width, height=args.height, n_per_team=args.n_per_team,
                               max_steps=args.max_steps, seed=args.seed, use_obstacles=True, obstacle_rate=0.06)
    team_obs_dim = env.get_team_obs_dim(); num_tactics = 4; action_dim = 10; n_units = env.n

    # ==== 모델 4개: A/B 팀 분리 + 유닛 멀티헤드 ====
    teamA = TeamTacticActorCritic(team_obs_dim, num_tactics).to(DEVICE)
    teamB = TeamTacticActorCritic(team_obs_dim, num_tactics).to(DEVICE)
    unitA = UnitActorCriticMultiHead(team_obs_dim, num_tactics, n_units, action_dim).to(DEVICE)
    unitB = UnitActorCriticMultiHead(team_obs_dim, num_tactics, n_units, action_dim).to(DEVICE)

    opt_teamA = optim.Adam(teamA.parameters(), lr=5e-4)
    opt_teamB = optim.Adam(teamB.parameters(), lr=5e-4)
    opt_unitA = optim.Adam(unitA.parameters(), lr=1e-3)
    opt_unitB = optim.Adam(unitB.parameters(), lr=1e-3)

    vf_coef = 0.5; ent_coef_team = 0.02; ent_coef_unit = 0.01; max_grad_norm = 0.5

    for ep in range(1, args.epochs + 1):
        buf = rollout(env, teamA, unitA, teamB, unitB, horizon=args.horizon, num_tactics=num_tactics)
        roll = buf.to_tensors()
        advA, tgtA, advB, tgtB = compute_advantages(roll, n_units=n_units)

        # ---- 팀 손실 (A/B 별도) ----
        logits_zA, v_team_A = teamA(roll["obs_team_A"])
        logits_zB, v_team_B = teamB(roll["obs_team_B"])
        pi_zA = Categorical(logits=logits_zA); pi_zB = Categorical(logits=logits_zB)
        with torch.no_grad():
            R_A_team = roll["rew_A"].flip(0).cumsum(0).flip(0)
            R_B_team = roll["rew_B"].flip(0).cumsum(0).flip(0)
        logp_zA = pi_zA.log_prob(roll["zA"]); logp_zB = pi_zB.log_prob(roll["zB"])
        adv_team_A = (R_A_team - v_team_A).detach(); adv_team_B = (R_B_team - v_team_B).detach()
        ent_team = (pi_zA.entropy().mean() + pi_zB.entropy().mean()) * 0.5
        loss_team = (-(logp_zA * adv_team_A).mean() + 0.5*((v_team_A-R_A_team)**2).mean()) + \
                    (-(logp_zB * adv_team_B).mean() + 0.5*((v_team_B-R_B_team)**2).mean()) - ent_coef_team * ent_team

        # ---- 유닛 손실 (A/B 별도) ----
        # 전술 원-핫 (B,T) -> (B,n,T) -> (B*n,T)
        zA_oh = torch.nn.functional.one_hot(roll["zA"], num_classes=num_tactics).float().unsqueeze(1).repeat(1, n_units, 1).reshape(-1, num_tactics)
        zB_oh = torch.nn.functional.one_hot(roll["zB"], num_classes=num_tactics).float().unsqueeze(1).repeat(1, n_units, 1).reshape(-1, num_tactics)
        logits_uA, vA = unitA(roll["unit_obs_A"], zA_oh, roll["unit_id_A"])
        logits_uB, vB = unitB(roll["unit_obs_B"], zB_oh, roll["unit_id_B"])
        pi_uA = Categorical(logits=logits_uA); pi_uB = Categorical(logits=logits_uB)
        ent_unit = (pi_uA.entropy().mean() + pi_uB.entropy().mean()) * 0.5
        logpA = pi_uA.log_prob(roll["act_A"]); logpB = pi_uB.log_prob(roll["act_B"])
        loss_unit = (-(logpA * advA).mean() + 0.5*((vA - tgtA)**2).mean()) + \
                    (-(logpB * advB).mean() + 0.5*((vB - tgtB)**2).mean()) - ent_coef_unit * ent_unit

        # ---- 업데이트 ----
        opt_teamA.zero_grad(); opt_teamB.zero_grad(); opt_unitA.zero_grad(); opt_unitB.zero_grad()
        (loss_team + loss_unit).backward()
        nn.utils.clip_grad_norm_(teamA.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(teamB.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(unitA.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(unitB.parameters(), max_grad_norm)
        opt_teamA.step(); opt_teamB.step(); opt_unitA.step(); opt_unitB.step()

        if ep % 10 == 0:
            with torch.no_grad():
                ep_rewA = roll["rew_A"].sum().item(); ep_rewB = roll["rew_B"].sum().item()
            print(f"ep {ep:5d} | A_reward={ep_rewA:+.2f} | B_reward={ep_rewB:+.2f} | teamL={loss_team.item():.3f} unitL={loss_unit.item():.3f}")

        # 엔트로피 감소 스케줄(탐험 -> 수렴)
        if ep == 300:
            ent_coef_team *= 0.5
            ent_coef_unit *= 0.5

        # ---- 저장 ----
        if args.save_every > 0 and ep % args.save_every == 0:
            meta = {"ep": ep, "width": env.width, "height": env.height, "n_per_team": env.n, "max_steps": env.max_steps}
            save_ckpt(os.path.join(args.save_dir, f"A_ep{ep}.pt"), teamA, unitA, meta)
            save_ckpt(os.path.join(args.save_dir, f"B_ep{ep}.pt"), teamB, unitB, meta)


if __name__ == "__main__":
    main()
