# train_ac_selfplay_5v5.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from multiagent_env_5v5 import CombatSelfPlay5v5Env, ACTION_NUM
from ac_model import TeamTacticActorCritic, UnitActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Utils
# ----------------------------
def make_unit_onehots(n, device):
    # shape: [n, n] (아이디 원-핫)
    return torch.eye(n, device=device)


def rollout(env, teamA, unitA, teamB, unitB, horizon, num_tactics, eps=0.0):
    """
    하나의 롤아웃을 수집. 팀 A/B 각각 별도 정책을 사용.
    eps>0면 유닛 행동에 ε-greedy를 섞어 탐험 유도.
    """
    buf = {
        "obsA": [], "obsB": [], "zA": [], "zB": [],
        "uobsA": [], "uobsB": [], "uidA": [], "uidB": [],
        "actA": [], "actB": [], "logpA": [], "logpB": [],
        "vA": [], "vB": [],
        "rA": [], "rB": [], "done": []
    }

    team_obs_A, team_obs_B = env.reset()
    n = env.n
    uid_eye = make_unit_onehots(n, DEVICE)  # [n, n]

    for _ in range(horizon):
        tA = torch.tensor(team_obs_A, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        tB = torch.tensor(team_obs_B, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # 팀 전술 샘플
        logits_zA, v_teamA = teamA(tA)
        logits_zB, v_teamB = teamB(tB)
        zA = Categorical(logits=logits_zA).sample().item()
        zB = Categorical(logits=logits_zB).sample().item()

        zA_oh = torch.nn.functional.one_hot(torch.tensor([zA], device=DEVICE),
                                            num_classes=num_tactics).float()
        zB_oh = torch.nn.functional.one_hot(torch.tensor([zB], device=DEVICE),
                                            num_classes=num_tactics).float()

        # 유닛 정책
        tA_rep = tA.repeat(n, 1)
        tB_rep = tB.repeat(n, 1)
        zA_rep = zA_oh.repeat(n, 1)
        zB_rep = zB_oh.repeat(n, 1)
        uidA = uid_eye
        uidB = uid_eye

        logits_uA, vA = unitA(tA_rep, zA_rep, uidA)
        logits_uB, vB = unitB(tB_rep, zB_rep, uidB)

        piA = Categorical(logits=logits_uA)
        piB = Categorical(logits=logits_uB)
        aA = piA.sample()
        aB = piB.sample()

        # ε-greedy
        if eps > 0.0:
            with torch.no_grad():
                randA = torch.rand_like(aA.float()) < eps
                randB = torch.rand_like(aB.float()) < eps
                if randA.any():
                    aA[randA] = torch.randint(0, ACTION_NUM, (int(randA.sum().item()),), device=aA.device)
                if randB.any():
                    aB[randB] = torch.randint(0, ACTION_NUM, (int(randB.sum().item()),), device=aB.device)

        logpA = piA.log_prob(aA)
        logpB = piB.log_prob(aB)

        # env step
        aA_np = aA.detach().cpu().numpy().astype(np.int32)
        aB_np = aB.detach().cpu().numpy().astype(np.int32)
        nextA, nextB, rA, rB, done, info = env.step(aA_np, aB_np)

        # 버퍼 적재 (학습 그래프는 다음 단계에서 재구성하므로 detach)
        buf["obsA"].append(tA.squeeze(0))
        buf["obsB"].append(tB.squeeze(0))
        buf["zA"].append(zA)
        buf["zB"].append(zB)
        buf["uobsA"].append(tA_rep.detach())
        buf["uobsB"].append(tB_rep.detach())
        buf["uidA"].append(uidA.detach())
        buf["uidB"].append(uidB.detach())
        buf["actA"].append(aA.detach())
        buf["actB"].append(aB.detach())
        buf["logpA"].append(logpA.detach())
        buf["logpB"].append(logpB.detach())
        buf["vA"].append(vA.detach())
        buf["vB"].append(vB.detach())
        buf["rA"].append(rA)
        buf["rB"].append(rB)
        buf["done"].append(1.0 if done else 0.0)

        team_obs_A, team_obs_B = nextA, nextB
        if done:
            team_obs_A, team_obs_B = env.reset()

    def cat(xs):
        return torch.cat(xs, dim=0) if isinstance(xs[0], torch.Tensor) else torch.tensor(xs)

    T = len(buf["rA"])
    roll = {
        "obsA": torch.stack(buf["obsA"]).to(DEVICE),
        "obsB": torch.stack(buf["obsB"]).to(DEVICE),
        "zA": torch.tensor(buf["zA"], dtype=torch.long, device=DEVICE),
        "zB": torch.tensor(buf["zB"], dtype=torch.long, device=DEVICE),
        "uobsA": cat(buf["uobsA"]).to(DEVICE),
        "uobsB": cat(buf["uobsB"]).to(DEVICE),
        "uidA": cat(buf["uidA"]).to(DEVICE),
        "uidB": cat(buf["uidB"]).to(DEVICE),
        "actA": cat(buf["actA"]).long().to(DEVICE),
        "actB": cat(buf["actB"]).long().to(DEVICE),
        "vA": cat(buf["vA"]).to(DEVICE),
        "vB": cat(buf["vB"]).to(DEVICE),
        "rA": torch.tensor(buf["rA"], dtype=torch.float32, device=DEVICE),
        "rB": torch.tensor(buf["rB"], dtype=torch.float32, device=DEVICE),
        "done": torch.tensor(buf["done"], dtype=torch.float32, device=DEVICE),
        "T": T
    }
    return roll


def compute_advantages(roll, gamma=0.99, n_units=10):
    T = roll["T"]
    dones = roll["done"]

    def returns(rews):
        ret = torch.zeros_like(rews)
        running = 0.0
        for t in reversed(range(T)):
            running = rews[t] + gamma * running * (1.0 - dones[t])
            ret[t] = running
        return ret

    R_A = returns(roll["rA"])
    R_B = returns(roll["rB"])

    vA = roll["vA"].view(T, n_units)
    vB = roll["vB"].view(T, n_units)
    tgtA = R_A.unsqueeze(1).repeat(1, n_units)
    tgtB = R_B.unsqueeze(1).repeat(1, n_units)

    advA = (tgtA - vA).detach()
    advB = (tgtB - vB).detach()
    return advA.reshape(-1), tgtA.reshape(-1), advB.reshape(-1), tgtB.reshape(-1)


def save_ckpt(path, team, unit, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"team": team.state_dict(), "unit": unit.state_dict(), "meta": meta}, path)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--height", type=int, default=24)
    ap.add_argument("--n_per_team", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--save_dir", type=str, default="ckpt")
    ap.add_argument("--save_every", type=int, default=100)
    # 탐험 스케줄
    ap.add_argument("--eps0", type=float, default=0.20)
    ap.add_argument("--eps_min", type=float, default=0.02)
    ap.add_argument("--eps_decay", type=int, default=800)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = CombatSelfPlay5v5Env(width=args.width, height=args.height,
                               n_per_team=args.n_per_team, max_steps=args.max_steps,
                               seed=args.seed, use_obstacles=True, obstacle_rate=0.06)

    obs_dim = env.get_team_obs_dim()
    num_tactics = 4
    action_dim = ACTION_NUM
    n_units = env.n

    # A/B 독립 정책
    teamA = TeamTacticActorCritic(obs_dim, num_tactics).to(DEVICE)
    unitA = UnitActorCritic(obs_dim, num_tactics, n_units, action_dim).to(DEVICE)
    teamB = TeamTacticActorCritic(obs_dim, num_tactics).to(DEVICE)
    unitB = UnitActorCritic(obs_dim, num_tactics, n_units, action_dim).to(DEVICE)

    opt_teamA = optim.Adam(teamA.parameters(), lr=3e-4)
    opt_unitA = optim.Adam(unitA.parameters(), lr=3e-4)
    opt_teamB = optim.Adam(teamB.parameters(), lr=3e-4)
    opt_unitB = optim.Adam(unitB.parameters(), lr=3e-4)

    vf_coef = 0.5
    ent_coef_team = 0.05
    ent_coef_unit = 0.02
    max_grad_norm = 0.5

    for ep in range(1, args.epochs + 1):
        # ε 스케줄
        eps = max(args.eps_min, args.eps0 * (1.0 - (ep - 1) / max(1, args.eps_decay)))

        roll = rollout(env, teamA, unitA, teamB, unitB,
                       horizon=args.horizon, num_tactics=num_tactics, eps=eps)
        advA, tgtA, advB, tgtB = compute_advantages(roll, n_units=n_units)

        # --- Team losses (A/B 분리 계산) ---
        logits_zA, v_teamA = teamA(roll["obsA"])
        logits_zB, v_teamB = teamB(roll["obsB"])
        pi_zA = Categorical(logits=logits_zA)
        pi_zB = Categorical(logits=logits_zB)

        with torch.no_grad():
            R_A_team = roll["rA"].flip(0).cumsum(0).flip(0)
            R_B_team = roll["rB"].flip(0).cumsum(0).flip(0)

        logp_zA = pi_zA.log_prob(roll["zA"])
        logp_zB = pi_zB.log_prob(roll["zB"])

        ent_teamA = pi_zA.entropy().mean()
        ent_teamB = pi_zB.entropy().mean()

        loss_teamA = -(logp_zA * (R_A_team - v_teamA).detach()).mean() \
                     + vf_coef * ((v_teamA - R_A_team) ** 2).mean() \
                     - ent_coef_team * ent_teamA

        loss_teamB = -(logp_zB * (R_B_team - v_teamB).detach()).mean() \
                     + vf_coef * ((v_teamB - R_B_team) ** 2).mean() \
                     - ent_coef_team * ent_teamB

        # --- Unit losses (A/B 분리 계산) ---
        zA_oh = torch.nn.functional.one_hot(roll["zA"], num_classes=num_tactics).float() \
                    .unsqueeze(1).repeat(1, n_units, 1).reshape(-1, num_tactics)
        zB_oh = torch.nn.functional.one_hot(roll["zB"], num_classes=num_tactics).float() \
                    .unsqueeze(1).repeat(1, n_units, 1).reshape(-1, num_tactics)

        logits_uA, vA = unitA(roll["uobsA"], zA_oh, roll["uidA"])
        logits_uB, vB = unitB(roll["uobsB"], zB_oh, roll["uidB"])
        pi_uA = Categorical(logits=logits_uA)
        pi_uB = Categorical(logits=logits_uB)

        ent_unitA = pi_uA.entropy().mean()
        ent_unitB = pi_uB.entropy().mean()

        loss_unitA = -(pi_uA.log_prob(roll["actA"]) * advA).mean() \
                     + vf_coef * ((vA - tgtA) ** 2).mean() \
                     - ent_coef_unit * ent_unitA

        loss_unitB = -(pi_uB.log_prob(roll["actB"]) * advB).mean() \
                     + vf_coef * ((vB - tgtB) ** 2).mean() \
                     - ent_coef_unit * ent_unitB

        # --- Optimize (A/B 각각 독립 그래프) ---
        opt_teamA.zero_grad();  loss_teamA.backward();  nn.utils.clip_grad_norm_(teamA.parameters(), max_grad_norm);  opt_teamA.step()
        opt_unitA.zero_grad();  loss_unitA.backward();  nn.utils.clip_grad_norm_(unitA.parameters(), max_grad_norm);  opt_unitA.step()

        opt_teamB.zero_grad();  loss_teamB.backward();  nn.utils.clip_grad_norm_(teamB.parameters(), max_grad_norm);  opt_teamB.step()
        opt_unitB.zero_grad();  loss_unitB.backward();  nn.utils.clip_grad_norm_(unitB.parameters(), max_grad_norm);  opt_unitB.step()

        if ep % 10 == 0:
            print(f"ep {ep:4d} | A_reward={roll['rA'].sum().item():+.2f} | B_reward={roll['rB'].sum().item():+.2f} "
                  f"| teamA_L={loss_teamA.item():.3f} unitA_L={loss_unitA.item():.3f}")

        # 엔트로피 점감(학습 진행에 따라 탐험 약화)
        if ep in (300, 800):
            ent_coef_team *= 0.5
            ent_coef_unit *= 0.5

        # 체크포인트 저장
        if args.save_every > 0 and ep % args.save_every == 0:
            meta = {"ep": ep, "width": env.width, "height": env.height,
                    "n_per_team": env.n, "max_steps": env.max_steps}
            save_ckpt(os.path.join(args.save_dir, f"A_ep{ep}.pt"), teamA, unitA, meta)
            save_ckpt(os.path.join(args.save_dir, f"B_ep{ep}.pt"), teamB, unitB, meta)


if __name__ == "__main__":
    main()
