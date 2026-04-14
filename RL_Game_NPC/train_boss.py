"""보스 레이드 PPO 학습 진입점

사용법:
    python train_boss.py --episodes 5000 --save-dir models_boss_v1

원 논문의 PPO 에이전트(src/agent.py)를 재활용한다.
플레이어 슬롯(딜러)은 학습 시 FSM 정책으로 대체하거나 v11 체크포인트 사용.
"""
import argparse
import os
import time
import numpy as np
import torch

from src.boss import BossConfig, BossRaidEnv, FSMNpcPolicy, PartyRole
from src.boss.config import BossActionID
from src.agent import ActorCritic
from src.config import TrainConfig


def dealer_fsm_action(env: BossRaidEnv) -> int:
    """플레이어 슬롯(딜러) 시뮬레이션용 간단 FSM.

    학습 데이터에서 플레이어 자리를 채우기 위한 더미.
    실제 실험에서는 Unity로부터 TCP로 입력을 받는다.
    """
    u = env.units[env.config.player_slot]
    if not u.alive:
        return int(BossActionID.STAY)
    # 위험 타일 회피
    danger = set()
    for tg in env.boss.telegraphs:
        danger |= tg.danger_tiles
    if (u.x, u.y) in danger:
        for act, dx, dy in (
            (BossActionID.MOVE_UP, 0, -1),
            (BossActionID.MOVE_DOWN, 0, 1),
            (BossActionID.MOVE_LEFT, -1, 0),
            (BossActionID.MOVE_RIGHT, 1, 0),
        ):
            nx, ny = u.x + dx, u.y + dy
            if (nx, ny) not in danger and 0 <= nx < env.config.map_width and 0 <= ny < env.config.map_height:
                return int(act)
    # 보스 공격
    if env._boss_dist(u.x, u.y) <= u.attack_range:
        if u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) <= 0:
            return int(BossActionID.ATTACK_SKILL)
        return int(BossActionID.ATTACK_BASIC)
    # 접근
    dx = env.boss.x - u.x; dy = env.boss.y - u.y
    if abs(dx) >= abs(dy):
        return int(BossActionID.MOVE_RIGHT) if dx > 0 else int(BossActionID.MOVE_LEFT)
    return int(BossActionID.MOVE_DOWN) if dy > 0 else int(BossActionID.MOVE_UP)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, default="models_boss_v1")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    cfg = BossConfig()
    env = BossRaidEnv(cfg)

    # NPC 3개(탱커, 힐러, 서포터)용 공유 PPO 네트워크
    obs_size = cfg.obs_size
    net = ActorCritic(obs_size=obs_size, action_size=cfg.num_actions, hidden_size=256).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # NPC 슬롯 (딜러 제외)
    npc_slots = [i for i, r in enumerate(cfg.party_roles) if r != PartyRole.DEALER]

    # 롤아웃 버퍼 (간단 버전)
    buf_obs, buf_act, buf_logp, buf_rew, buf_val, buf_done = [], [], [], [], [], []

    ep_rewards = []
    ep_steps = []
    victories = 0
    start = time.time()

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        ep_rew = 0.0
        ep_step = 0
        done = False

        while not done:
            actions = {}

            # 딜러 (FSM)
            actions[f"p{cfg.player_slot}"] = dealer_fsm_action(env)

            # NPC (PPO)
            step_transitions = []
            for uid in npc_slots:
                aid = f"p{uid}"
                o = torch.as_tensor(obs[aid], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, value = net.get_action(o)
                a = int(action.item())
                actions[aid] = a
                step_transitions.append((aid, obs[aid], a, log_prob.item(), value.item()))

            obs_next, rewards, dones, infos = env.step(actions)

            for aid, o, a, lp, v in step_transitions:
                buf_obs.append(o)
                buf_act.append(a)
                buf_logp.append(lp)
                buf_val.append(v)
                buf_rew.append(rewards[aid])
                buf_done.append(dones[aid])

            ep_rew += sum(rewards[f"p{uid}"] for uid in npc_slots) / len(npc_slots)
            ep_step += 1
            done = all(dones.values())
            obs = obs_next

        ep_rewards.append(ep_rew)
        ep_steps.append(ep_step)
        if env.victory:
            victories += 1

        # PPO 업데이트 (간이: 매 에피소드마다)
        if len(buf_obs) >= 256:
            _ppo_update(net, optim, buf_obs, buf_act, buf_logp, buf_rew, buf_val, buf_done, device)
            buf_obs.clear(); buf_act.clear(); buf_logp.clear()
            buf_rew.clear(); buf_val.clear(); buf_done.clear()

        if episode % args.log_interval == 0:
            avg_r = np.mean(ep_rewards[-args.log_interval:])
            avg_s = np.mean(ep_steps[-args.log_interval:])
            wr = victories / episode
            elapsed = time.time() - start
            print(f"[EP {episode:5d}] R={avg_r:+.2f}  steps={avg_s:.1f}  "
                  f"victory_rate={wr*100:.1f}%  elapsed={elapsed:.0f}s")

        if episode % args.save_interval == 0:
            path = os.path.join(args.save_dir, f"ckpt_{episode}.pt")
            torch.save({"net": net.state_dict(), "episode": episode}, path)
            print(f"  ↳ saved {path}")

    # 최종 저장
    torch.save({"net": net.state_dict(), "episode": args.episodes},
               os.path.join(args.save_dir, "final.pt"))
    print("Training done.")


def _ppo_update(net, optim, obs, act, old_logp, rew, val, done, device,
                gamma=0.99, lam=0.95, clip=0.2, epochs=4, batch=64):
    import torch.nn.functional as F
    obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device)
    act = torch.as_tensor(act, dtype=torch.long, device=device)
    old_logp = torch.as_tensor(old_logp, dtype=torch.float32, device=device)
    rew = np.array(rew, dtype=np.float32)
    val = np.array(val, dtype=np.float32)
    done = np.array(done, dtype=np.float32)

    # GAE
    adv = np.zeros_like(rew)
    last = 0.0
    for t in reversed(range(len(rew))):
        next_v = 0 if done[t] else (val[t + 1] if t + 1 < len(val) else 0)
        delta = rew[t] + gamma * next_v * (1 - done[t]) - val[t]
        last = delta + gamma * lam * (1 - done[t]) * last
        adv[t] = last
    ret = adv + val
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    adv = torch.as_tensor(adv, dtype=torch.float32, device=device)
    ret = torch.as_tensor(ret, dtype=torch.float32, device=device)

    N = len(obs)
    idx = np.arange(N)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, N, batch):
            b = idx[start:start + batch]
            logits, v = net(obs[b])
            probs = F.softmax(logits, dim=-1)
            logp = torch.log(probs.gather(1, act[b].unsqueeze(1)).squeeze(1) + 1e-8)
            ratio = torch.exp(logp - old_logp[b])
            s1 = ratio * adv[b]
            s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = F.mse_loss(v.squeeze(-1), ret[b])
            ent = -(probs * torch.log(probs + 1e-8)).sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * ent
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optim.step()


if __name__ == "__main__":
    main()
