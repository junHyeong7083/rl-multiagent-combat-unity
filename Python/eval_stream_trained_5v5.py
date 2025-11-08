import argparse
import json
import socket
import time
import torch
import numpy as np
from torch.distributions import Categorical

from multiagent_env_5v5 import CombatSelfPlay5v5Env, ACTION_NUM
from ac_model import TeamTacticActorCritic, UnitActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pair(ckpt_team_path, ckpt_unit_path, obs_dim, num_tactics, n_units, action_dim):
    c1 = torch.load(ckpt_team_path, map_location=DEVICE)
    c2 = torch.load(ckpt_unit_path, map_location=DEVICE)

    team = TeamTacticActorCritic(obs_dim, num_tactics).to(DEVICE)
    unit = UnitActorCritic(obs_dim, num_tactics, n_units, action_dim).to(DEVICE)
    team.load_state_dict(c1["team"] if "team" in c1 else c1["state_dict"], strict=False)
    unit.load_state_dict(c2["unit"] if "unit" in c2 else c2["state_dict"], strict=False)
    team.eval(); unit.eval()
    return team, unit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--fps", type=int, default=10)

    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--height", type=int, default=24)
    ap.add_argument("--n_per_team", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--obstacle_rate", type=float, default=0.06)

    ap.add_argument("--A_team_ckpt", type=str, required=True)
    ap.add_argument("--A_unit_ckpt", type=str, required=True)
    ap.add_argument("--B_team_ckpt", type=str, required=True)
    ap.add_argument("--B_unit_ckpt", type=str, required=True)

    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    env = CombatSelfPlay5v5Env(width=args.width, height=args.height, n_per_team=args.n_per_team,
                               max_steps=args.max_steps, seed=args.seed,
                               use_obstacles=args.obstacles, obstacle_rate=args.obstacle_rate)

    obs_dim = env.get_team_obs_dim()
    num_tactics = 4
    action_dim = ACTION_NUM
    n_units = env.n

    teamA, unitA = load_pair(args.A_team_ckpt, args.A_unit_ckpt, obs_dim, num_tactics, n_units, action_dim)
    teamB, unitB = load_pair(args.B_team_ckpt, args.B_unit_ckpt, obs_dim, num_tactics, n_units, action_dim)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    dt = 1.0 / max(1, args.fps)
    print(f"[eval_AB] UDP {addr} / {args.fps}FPS det={args.deterministic} eps={args.eps} temp={args.temp}")

    A_obs, B_obs = env.reset()
    t0 = time.time()
    while True:
        tA = torch.tensor(A_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        tB = torch.tensor(B_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # 전술
        logits_zA, _ = teamA(tA)
        logits_zB, _ = teamB(tB)
        if args.temp != 1.0:
            logits_zA = logits_zA / args.temp
            logits_zB = logits_zB / args.temp
        if args.deterministic:
            zA = torch.argmax(logits_zA, dim=-1).item()
            zB = torch.argmax(logits_zB, dim=-1).item()
        else:
            zA = Categorical(logits=logits_zA).sample().item()
            zB = Categorical(logits=logits_zB).sample().item()

        zA_oh = torch.nn.functional.one_hot(torch.tensor([zA], device=DEVICE), num_classes=4).float()
        zB_oh = torch.nn.functional.one_hot(torch.tensor([zB], device=DEVICE), num_classes=4).float()
        n = env.n
        tA_rep = tA.repeat(n,1); tB_rep = tB.repeat(n,1)
        zA_rep = zA_oh.repeat(n,1); zB_rep = zB_oh.repeat(n,1)
        uid = torch.eye(n, device=DEVICE)

        logits_uA, _ = unitA(tA_rep, zA_rep, uid)
        logits_uB, _ = unitB(tB_rep, zB_rep, uid)

        if args.temp != 1.0:
            logits_uA = logits_uA / args.temp
            logits_uB = logits_uB / args.temp

        if args.deterministic:
            aA = torch.argmax(logits_uA, dim=-1)
            aB = torch.argmax(logits_uB, dim=-1)
        else:
            aA = Categorical(logits=logits_uA).sample()
            aB = Categorical(logits=logits_uB).sample()
            if args.eps > 0.0:
                with torch.no_grad():
                    randA = torch.rand_like(aA.float()) < args.eps
                    randB = torch.rand_like(aB.float()) < args.eps
                    aA[randA] = torch.randint(0, ACTION_NUM, (randA.sum().item(),), device=aA.device)
                    aB[randB] = torch.randint(0, ACTION_NUM, (randB.sum().item(),), device=aB.device)

        A_obs, B_obs, rA, rB, done, info = env.step(aA.detach().cpu().numpy().astype(np.int32),
                                                     aB.detach().cpu().numpy().astype(np.int32))

        frame = env.make_frame()
        # 보상/종료 결과도 넣어주면 디버그에 유용
        frame["rA"] = rA; frame["rB"] = rB
        frame["outcome"] = info.get("outcome", None)

        payload = json.dumps(frame).encode("utf-8")
        sock.sendto(payload, addr)

        if done:
            A_obs, B_obs = env.reset()

        # FPS 맞추기
        d = t0 + dt - time.time()
        if d > 0: time.sleep(d)
        t0 = time.time()

if __name__ == "__main__":
    main()
