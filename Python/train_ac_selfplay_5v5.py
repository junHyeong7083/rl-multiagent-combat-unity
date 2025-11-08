# train_ac_selfplay_5v5.py
import os, argparse, torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from multiagent_env_5v5 import CombatSelfPlay5v5Env, ACTION_NUM
from ac_model import TeamTacticActorCritic, UnitActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eye_n(n): 
    return torch.eye(n, device=DEVICE)

def rollout(env, teamA, unitA, teamB, unitB, horizon, num_tactics, eps=0.05, temp=1.0):
    """
    계층형 롤아웃: 팀 전술(분포)에서 유닛별 z 샘플 → 유닛 자율/지시 혼합 로짓
    그래프 유지를 위해 모델 출력 텐서들에 detach() 금지!
    """
    buf = {k: [] for k in [
        # 팀 관측/전술
        "obsA","obsB","zA_vec","zB_vec","logp_zA","logp_zB","v_teamA","v_teamB","ent_zA","ent_zB",
        # 유닛
        "ulogsA","ulogsB","logpA","logpB","vA","vB","alphaA","alphaB","localA","localB","priorA","priorB",
        "actA","actB",
        # 보상/종료
        "rA","rB","done"
    ]}
    obsA, obsB = env.reset()
    n = env.n
    uid_eye = eye_n(n)

    for _ in range(horizon):
        tA = torch.tensor(obsA, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, obs]
        tB = torch.tensor(obsB, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, obs]

        # 팀 전술 분포
        logits_zA, v_tA = teamA(tA)            # logits [1,Z], value [1,1] or [1]
        logits_zB, v_tB = teamB(tB)
        pi_zA = Categorical(logits=logits_zA / max(1e-8, temp))
        pi_zB = Categorical(logits=logits_zB / max(1e-8, temp))

        # 유닛별 전술 샘플 (sample_shape=n → [n,1])
        zA_vec_raw = pi_zA.sample((n,))   # [n, 1]
        zB_vec_raw = pi_zB.sample((n,))   # [n, 1]
        zA_vec = zA_vec_raw.squeeze(-1)   # [n]
        zB_vec = zB_vec_raw.squeeze(-1)   # [n]

        zA_oh  = F.one_hot(zA_vec, num_classes=num_tactics).float()  # [n,Z]
        zB_oh  = F.one_hot(zB_vec, num_classes=num_tactics).float()  # [n,Z]

        # 유닛 정책 (자율+prior 혼합 로짓) — UnitActorCritic는 팀 관측 [B,obs] 입력
        ulogitsA, vA, alphaA, localA, priorA = unitA(tA, zA_oh, uid_eye)  # [n,A], [n], [n], [n,A], [n,A]
        ulogitsB, vB, alphaB, localB, priorB = unitB(tB, zB_oh, uid_eye)

        # 샘플링 (+ ε-greedy)
        piA = Categorical(logits=ulogitsA)
        piB = Categorical(logits=ulogitsB)
        aA = piA.sample()  # [n]
        aB = piB.sample()  # [n]

        if eps > 0.0:
            with torch.no_grad():
                maskA = (torch.rand(n, device=DEVICE) < eps)
                maskB = (torch.rand(n, device=DEVICE) < eps)
                if maskA.any():
                    aA[maskA] = torch.randint(0, ACTION_NUM, (int(maskA.sum().item()),), device=DEVICE)
                if maskB.any():
                    aB[maskB] = torch.randint(0, ACTION_NUM, (int(maskB.sum().item()),), device=DEVICE)

        # 환경 step (환경 인자로만 numpy 변환)
        obsA, obsB, rA, rB, done, _info = env.step(aA.cpu().numpy(), aB.cpu().numpy())

        # ==== 버퍼 적재 (detach 금지) ====
        # 팀
        buf["obsA"].append(tA.squeeze(0))                              # [obs]
        buf["obsB"].append(tB.squeeze(0))
        buf["zA_vec"].append(zA_vec)                                   # [n]
        buf["zB_vec"].append(zB_vec)
        buf["logp_zA"].append(pi_zA.log_prob(zA_vec_raw).mean())       # scalar
        buf["logp_zB"].append(pi_zB.log_prob(zB_vec_raw).mean())       # scalar
        buf["v_teamA"].append(v_tA.squeeze())                           # scalar
        buf["v_teamB"].append(v_tB.squeeze())                           # scalar
        buf["ent_zA"].append(pi_zA.entropy().squeeze())                # scalar
        buf["ent_zB"].append(pi_zB.entropy().squeeze())                # scalar

        # 유닛 (모델 출력, logp 모두 그래프 유지)
        buf["ulogsA"].append(ulogitsA)                                  # [n,A]
        buf["ulogsB"].append(ulogitsB)
        buf["logpA"].append(piA.log_prob(aA))                           # [n]
        buf["logpB"].append(piB.log_prob(aB))                           # [n]
        buf["vA"].append(vA)                                            # [n]
        buf["vB"].append(vB)                                            # [n]
        buf["alphaA"].append(alphaA)                                    # [n]
        buf["alphaB"].append(alphaB)                                    # [n]
        buf["localA"].append(localA)                                    # [n,A]
        buf["localB"].append(localB)
        buf["priorA"].append(priorA)                                    # [n,A]
        buf["priorB"].append(priorB)
        buf["actA"].append(aA)                                          # [n] (정수, grad 불필요)
        buf["actB"].append(aB)

        # 보상/종료
        buf["rA"].append(torch.tensor(rA, dtype=torch.float32, device=DEVICE))  # scalar
        buf["rB"].append(torch.tensor(rB, dtype=torch.float32, device=DEVICE))  # scalar
        buf["done"].append(1.0 if done else 0.0)

        if done:
            obsA, obsB = env.reset()

    # === 스택/패킹 ===
    def Stack(x): return torch.stack(x)
    T = len(buf["rA"])
    roll = dict(
        T=T, n=n,
        # 팀
        obsA=Stack(buf["obsA"]),                    # [T, obs]
        obsB=Stack(buf["obsB"]),
        zA_vec=Stack(buf["zA_vec"]),                # [T, n]
        zB_vec=Stack(buf["zB_vec"]),
        logp_zA=Stack(buf["logp_zA"]),              # [T]
        logp_zB=Stack(buf["logp_zB"]),              # [T]
        v_teamA=Stack(buf["v_teamA"]),              # [T]
        v_teamB=Stack(buf["v_teamB"]),              # [T]
        ent_zA=Stack(buf["ent_zA"]),                # [T]
        ent_zB=Stack(buf["ent_zB"]),                # [T]
        # 유닛
        ulogsA=Stack(buf["ulogsA"]),                # [T, n, A]
        ulogsB=Stack(buf["ulogsB"]),                # [T, n, A]
        logpA=Stack(buf["logpA"]),                  # [T, n]
        logpB=Stack(buf["logpB"]),                  # [T, n]
        vA=Stack(buf["vA"]),                        # [T, n]
        vB=Stack(buf["vB"]),                        # [T, n]
        alphaA=Stack(buf["alphaA"]),                # [T, n]
        alphaB=Stack(buf["alphaB"]),                # [T, n]
        localA=Stack(buf["localA"]),                # [T, n, A]
        localB=Stack(buf["localB"]),                # [T, n, A]
        priorA=Stack(buf["priorA"]),                # [T, n, A]
        priorB=Stack(buf["priorB"]),                # [T, n, A]
        actA=Stack(buf["actA"]),                    # [T, n]
        actB=Stack(buf["actB"]),                    # [T, n]
        # 리워드/종료
        rA=Stack(buf["rA"]),                        # [T]
        rB=Stack(buf["rB"]),                        # [T]
        done=torch.tensor(buf["done"], dtype=torch.float32, device=DEVICE)  # [T]
    )
    return roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=200_000)
    ap.add_argument("--horizon", type=int, default=256)
    ap.add_argument("--num-tactics", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--ckpt-dir", type=str, default="./ckpt")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    env = CombatSelfPlay5v5Env()
    obs_dim = env.get_team_obs_dim()
    n_units = env.n
    action_dim = ACTION_NUM
    num_tactics = args.num_tactics
    gamma = args.gamma

    teamA = TeamTacticActorCritic(obs_dim, num_tactics).to(DEVICE)
    teamB = TeamTacticActorCritic(obs_dim, num_tactics).to(DEVICE)
    unitA = UnitActorCritic(obs_dim, num_tactics, n_units, action_dim).to(DEVICE)
    unitB = UnitActorCritic(obs_dim, num_tactics, n_units, action_dim).to(DEVICE)

    opt_teamA = optim.Adam(teamA.parameters(), lr=args.lr)
    opt_teamB = optim.Adam(teamB.parameters(), lr=args.lr)
    opt_unitA = optim.Adam(unitA.parameters(), lr=args.lr)
    opt_unitB = optim.Adam(unitB.parameters(), lr=args.lr)

    ent_coef_team = 0.01
    ent_coef_unit = 0.01
    vf_coef = 0.5
    kl_coef = 0.02  # prior 정합 가중치(작게 시작)

    step = 0
    while step < args.total_steps:
        roll = rollout(env, teamA, unitA, teamB, unitB,
                       args.horizon, num_tactics, eps=args.eps, temp=args.temp)
        T, n = roll["T"], roll["n"]
        step += T

        # ===== 팀 레벨 리턴/손실 =====
        with torch.no_grad():
            R_A_team = torch.zeros_like(roll["rA"])
            R_B_team = torch.zeros_like(roll["rB"])
            runA = 0.0; runB = 0.0
            for t in reversed(range(T)):
                runA = roll["rA"][t] + gamma * runA * (1.0 - roll["done"][t])
                runB = roll["rB"][t] + gamma * runB * (1.0 - roll["done"][t])
                R_A_team[t] = runA; R_B_team[t] = runB

        ent_teamA = roll["ent_zA"].mean()
        ent_teamB = roll["ent_zB"].mean()

        # Advantage는 target - value, policy-gradient는 logp * adv(target) (value는 detach X)
        loss_teamA = -(roll["logp_zA"] * (R_A_team - roll["v_teamA"]).detach()).mean() \
                     + vf_coef * ((roll["v_teamA"] - R_A_team)**2).mean() \
                     - ent_coef_team * ent_teamA

        loss_teamB = -(roll["logp_zB"] * (R_B_team - roll["v_teamB"]).detach()).mean() \
                     + vf_coef * ((roll["v_teamB"] - R_B_team)**2).mean() \
                     - ent_coef_team * ent_teamB

        # ===== 유닛 레벨 =====
        def returns(rews, done):
            ret = torch.zeros_like(rews); running = 0.0
            for t in reversed(range(T)):
                running = rews[t] + gamma * running * (1.0 - done[t])
                ret[t] = running
            return ret

        R_A = returns(roll["rA"], roll["done"]).unsqueeze(1).repeat(1, n)  # [T,n]
        R_B = returns(roll["rB"], roll["done"]).unsqueeze(1).repeat(1, n)  # [T,n]

        vA = roll["vA"]                # [T,n]
        vB = roll["vB"]                # [T,n]
        advA = R_A - vA
        advB = R_B - vB

        logpA = roll["logpA"]          # [T,n]
        logpB = roll["logpB"]          # [T,n]

        # 엔트로피: 저장된 logits로 재계산 (그래프 유지)
        ulogsA_flat = roll["ulogsA"].reshape(T*n, -1)
        ulogsB_flat = roll["ulogsB"].reshape(T*n, -1)
        entA = Categorical(logits=ulogsA_flat).entropy().mean()
        entB = Categorical(logits=ulogsB_flat).entropy().mean()

        loss_unitA = -(logpA * advA.detach()).mean() + 0.5 * (advA**2).mean() - ent_coef_unit * entA
        loss_unitB = -(logpB * advB.detach()).mean() + 0.5 * (advB**2).mean() - ent_coef_unit * entB

        # prior 정합 KL (α 가중)
        localA = roll["localA"].reshape(T*n, -1)
        priorA = roll["priorA"].reshape(T*n, -1)
        alphaA = roll["alphaA"].reshape(T*n)

        localB = roll["localB"].reshape(T*n, -1)
        priorB = roll["priorB"].reshape(T*n, -1)
        alphaB = roll["alphaB"].reshape(T*n)

        p_locA = F.softmax(localA, dim=-1)
        p_priA = F.softmax(priorA, dim=-1)
        p_locB = F.softmax(localB, dim=-1)
        p_priB = F.softmax(priorB, dim=-1)

        klA = torch.sum(p_locA * (torch.log(p_locA + 1e-8) - torch.log(p_priA + 1e-8)), dim=-1)  # [T*n]
        klB = torch.sum(p_locB * (torch.log(p_locB + 1e-8) - torch.log(p_priB + 1e-8)), dim=-1)

        loss_unitA = loss_unitA + kl_coef * (alphaA * klA).mean()
        loss_unitB = loss_unitB + kl_coef * (alphaB * klB).mean()

        # ===== 최적화 =====
        opt_teamA.zero_grad(); loss_teamA.backward(); opt_teamA.step()
        opt_teamB.zero_grad(); loss_teamB.backward(); opt_teamB.step()
        opt_unitA.zero_grad(); loss_unitA.backward(); opt_unitA.step()
        opt_unitB.zero_grad(); loss_unitB.backward(); opt_unitB.step()

        if step % args.save_every == 0:
            torch.save({"team": teamA.state_dict()}, os.path.join(args.ckpt_dir, f"teamA_{step}.pt"))
            torch.save({"team": teamB.state_dict()}, os.path.join(args.ckpt_dir, f"teamB_{step}.pt"))
            torch.save({"unit": unitA.state_dict()}, os.path.join(args.ckpt_dir, f"unitA_{step}.pt"))
            torch.save({"unit": unitB.state_dict()}, os.path.join(args.ckpt_dir, f"unitB_{step}.pt"))
            print(f"[{step}] teamA={loss_teamA.item():.3f} teamB={loss_teamB.item():.3f} "
                  f"unitA={loss_unitA.item():.3f} unitB={loss_unitB.item():.3f}")

if __name__ == "__main__":
    main()
