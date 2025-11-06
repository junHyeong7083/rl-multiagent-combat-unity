import os, json
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ac_model import ActorCritic  # 그대로 사용
from multiagent_env_5v5 import CombatSelfPlay5v5Env  # 그대로 사용

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 환경 & 모델 파라미터 =====
env = CombatSelfPlay5v5Env(width=12, height=8, n_per_team=5, max_steps=120, hp=3, reload_steps=5, seed=42)
state_dim = env._obs_team(env.A, env.B).shape[0]  # 중앙집중 관측(양 팀 정보 모두)
action_dim_per_agent = 10  # 0..9
n_agents = env.n

# 팀당 하나의 중앙정책(공유) -> 각 에이전트에 대해 독립적으로 10-way action을 n_agents개 샘플
model_A = ActorCritic(state_dim, action_dim_per_agent).to(device)
model_B = ActorCritic(state_dim, action_dim_per_agent).to(device)

opt_A = optim.Adam(model_A.parameters(), lr=1e-3)
opt_B = optim.Adam(model_B.parameters(), lr=1e-3)

gamma = 0.99
episodes = 10000          # ★ 500 -> 10000
entropy_coef = 0.01
grad_clip = 1.0           # ★ 그래디언트 클리핑
save_every = 1000         # ★ 체크포인트 저장 주기
replay_dir = "replays"    # ★ 리플레이 저장 폴더
ckpt_dir = "checkpoints"  # ★ 체크포인트 저장 폴더
os.makedirs(replay_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

def select_actions(model, obs_np, n_agents):
    x = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    logits, value = model(x)                              # (1, action_dim), (1,1)
    probs = F.softmax(logits, dim=-1).squeeze(0)          # (action_dim,)
    dist = torch.distributions.Categorical(probs)
    # n_agents개 샘플
    actions = dist.sample((n_agents,))                    # (n_agents,)
    logps = dist.log_prob(actions)                        # (n_agents,)
    ent = dist.entropy()                                  # scalar
    return actions.cpu().numpy().astype(np.int32), logps, value.squeeze(0), ent

for ep in range(episodes):
    obs_A, obs_B = env.reset()
    obs_A = obs_A.astype(np.float32)
    obs_B = obs_B.astype(np.float32)

    done = False
    logps_A, vals_A, rews_A, ents_A = [], [], [], []
    logps_B, vals_B, rews_B, ents_B = [], [], [], []

    # ★ 리플레이(스텝별 상태 저장)
    traj = []

    # ★ 스텝 로그: ep 0,1은 항상 / 그 이후는 100에피소드마다
    want_step_log = (ep < 2) or (ep % 100 == 0)

    step_idx = 0
    while not done:
        acts_A, logp_A, val_A, ent_A = select_actions(model_A, obs_A, n_agents)
        acts_B, logp_B, val_B, ent_B = select_actions(model_B, obs_B, n_agents)

        next_A, next_B, rA, rB, done, info = env.step(acts_A, acts_B)
        next_A = next_A.astype(np.float32)
        next_B = next_B.astype(np.float32)

        if want_step_log:
            print(f"[ep {ep} step {step_idx}] rA={rA:.2f} rB={rB:.2f}  "
                  f"A_alive={int((env.A[:,2]>0).sum())}  B_alive={int((env.B[:,2]>0).sum())}")

        # 팀 단위 학습 값 저장
        logps_A.append(logp_A.sum())
        vals_A.append(val_A)
        rews_A.append(torch.tensor(rA, dtype=torch.float32, device=device))
        ents_A.append(ent_A)

        logps_B.append(logp_B.sum())
        vals_B.append(val_B)
        rews_B.append(torch.tensor(rB, dtype=torch.float32, device=device))
        ents_B.append(ent_B)

        # ★ 리플레이 기록 (현재 상태를 기록)
        traj.append({
            "t": step_idx,
            "A": env.A.tolist(),  # [[x,y,hp,fx,fy,cd]*5]
            "B": env.B.tolist(),  # [[x,y,hp,fx,fy,cd]*5]
            "acts_A": acts_A.tolist(),
            "acts_B": acts_B.tolist(),
            "rA": float(rA),
            "rB": float(rB)
        })

        obs_A, obs_B = next_A, next_B
        step_idx += 1

    # ----- 업데이트 (A) -----
    returns_A = []
    G = torch.tensor(0.0, device=device)
    for r in reversed(rews_A):
        G = r + gamma * G
        returns_A.insert(0, G)
    returns_A = torch.stack(returns_A)

    vals_A = torch.stack(vals_A).view(-1)
    logps_A = torch.stack(logps_A).view(-1)
    ents_A = torch.stack(ents_A).view(-1)

    adv_A = returns_A - vals_A.detach()
    policy_loss_A = -(logps_A * adv_A).mean()
    value_loss_A = 0.5 * F.mse_loss(vals_A, returns_A)
    entropy_loss_A = ents_A.mean()

    loss_A = policy_loss_A + value_loss_A - entropy_coef * entropy_loss_A
    opt_A.zero_grad()
    loss_A.backward()
    # ★ 그래디언트 클리핑
    torch.nn.utils.clip_grad_norm_(model_A.parameters(), grad_clip)
    opt_A.step()

    # ----- 업데이트 (B) -----
    returns_B = []
    G = torch.tensor(0.0, device=device)
    for r in reversed(rews_B):
        G = r + gamma * G
        returns_B.insert(0, G)
    returns_B = torch.stack(returns_B)

    vals_B = torch.stack(vals_B).view(-1)
    logps_B = torch.stack(logps_B).view(-1)
    ents_B = torch.stack(ents_B).view(-1)

    adv_B = returns_B - vals_B.detach()
    policy_loss_B = -(logps_B * adv_B).mean()
    value_loss_B = 0.5 * F.mse_loss(vals_B, returns_B)
    entropy_loss_B = ents_B.mean()

    loss_B = policy_loss_B + value_loss_B - entropy_coef * entropy_loss_B
    opt_B.zero_grad()
    loss_B.backward()
    # ★ 그래디언트 클리핑
    torch.nn.utils.clip_grad_norm_(model_B.parameters(), grad_clip)
    opt_B.step()

    # ★ 리플레이 저장 (유니티 재생용)
    with open(os.path.join(replay_dir, f"ep_{ep:05d}.json"), "w", encoding="utf-8") as f:
        json.dump(traj, f, ensure_ascii=False)

    # ★ 에피소드 요약 로그: 100에피소드마다
    if (ep + 1) % 100 == 0:
        sum_rA = sum(r.item() for r in rews_A)
        sum_rB = sum(r.item() for r in rews_B)
        print(f"[summary] ep {ep+1} | A_reward={sum_rA:.2f} | B_reward={sum_rB:.2f}")

    # ★ 체크포인트 저장: 1000에피소드마다
    if (ep + 1) % save_every == 0:
        torch.save(model_A.state_dict(), os.path.join(ckpt_dir, f"model_A_ep{ep+1}.pt"))
        torch.save(model_B.state_dict(), os.path.join(ckpt_dir, f"model_B_ep{ep+1}.pt"))
        print(f"[ckpt] saved at ep {ep+1}")
