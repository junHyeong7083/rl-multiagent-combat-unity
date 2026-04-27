"""보스 레이드 PPO 학습 진입점

사용법:
    python train_boss.py --episodes 5000 --save-dir models_boss_v1

원 논문의 PPO 에이전트(src/agent.py)를 재활용한다.
플레이어 슬롯(딜러)은 학습 시 FSM 정책으로 대체하거나 v11 체크포인트 사용.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

# Windows 콘솔 버퍼링 방지 — 매 print가 즉시 출력되도록
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # Python 3.6 이하는 reconfigure 없음

from src.boss import BossConfig, BossRaidEnv, FSMNpcPolicy, PartyRole, BTPolicy
from src.boss.config import BossActionID, PatternID
from src.agent import ActorCritic
from src.config import TrainConfig


def dealer_fsm_action(env: BossRaidEnv) -> int:
    """플레이어 슬롯(딜러) 시뮬레이션용 FSM — **생존 우선** (유클리드).

    우선순위:
      1. 위험 영역 위: 무조건 회피
      2. HP < 40%: 보스에서 멀어짐 (후퇴)
      3. 어그로 1위 = 나: 탱커 쪽으로 이동 (어그로 돌려주기 대기)
      4. 사거리 내: 공격 (스킬 먼저)
      5. 사거리 밖: 접근 (탱커가 어그로 잡은 뒤에만)
    """
    import math as _m
    u = env.units[env.config.player_slot]
    if not u.alive:
        return int(BossActionID.STAY)

    # 0. SEAL_BREAK 활성 — 딜러는 자기 배정 spot으로 이동/유지
    for tg in env.boss.telegraphs:
        if tg.pattern_id != PatternID.SEAL_BREAK:
            continue
        dealer_spot_idx = tg.target_unit_ids[0] if tg.target_unit_ids else -1
        if 0 <= dealer_spot_idx < len(tg.shapes):
            s = tg.shapes[dealer_spot_idx]
            sx, sy = s.params["cx"], s.params["cy"]
            if _m.hypot(u.x - sx, u.y - sy) <= s.params["r"]:
                # 내 spot 위 — 보스가 사거리 안이면 때리고, 아니면 자리 사수
                if env._boss_dist(u.x, u.y) <= u.attack_range:
                    if u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) <= 0:
                        return int(BossActionID.ATTACK_SKILL)
                    return int(BossActionID.ATTACK_BASIC)
                return int(BossActionID.STAY)
            dx = sx - u.x; dy = sy - u.y
            if abs(dx) >= abs(dy):
                return int(BossActionID.MOVE_RIGHT) if dx > 0 else int(BossActionID.MOVE_LEFT)
            return int(BossActionID.MOVE_DOWN) if dy > 0 else int(BossActionID.MOVE_UP)
        break

    # 1. 위험 영역 회피 (SEAL_BREAK 제외)
    in_danger = any(
        tg.contains((u.x, u.y))
        for tg in env.boss.telegraphs
        if tg.pattern_id != PatternID.SEAL_BREAK
    )
    if in_danger:
        best_act = int(BossActionID.STAY)
        best_score = -1e9
        d = u.move_speed * 0.7071
        for act, dx, dy in (
            (BossActionID.MOVE_UP, 0, -u.move_speed),
            (BossActionID.MOVE_DOWN, 0, u.move_speed),
            (BossActionID.MOVE_LEFT, -u.move_speed, 0),
            (BossActionID.MOVE_RIGHT, u.move_speed, 0),
            (BossActionID.MOVE_UP_LEFT, -d, -d),
            (BossActionID.MOVE_UP_RIGHT, d, -d),
            (BossActionID.MOVE_DOWN_LEFT, -d, d),
            (BossActionID.MOVE_DOWN_RIGHT, d, d),
        ):
            nx, ny = u.x + dx, u.y + dy
            if not (u.radius <= nx <= env.config.map_width - u.radius and
                    u.radius <= ny <= env.config.map_height - u.radius):
                continue
            # 위험 영역 밖인지 + 얼마나 안전한지 점수화 (SEAL_BREAK 제외)
            still_dangerous = any(
                tg.contains((nx, ny))
                for tg in env.boss.telegraphs
                if tg.pattern_id != PatternID.SEAL_BREAK
            )
            score = 0 if still_dangerous else 10
            best_score = max(best_score, score)
            if score > 0: return int(act)
        # 모든 방향이 위험하면 대기
        return int(BossActionID.STAY)

    # 2. HP 낮으면 후퇴 (40% 이하)
    if u.hp / max(1, u.max_hp) < 0.4:
        dx = u.x - env.boss.x; dy = u.y - env.boss.y
        if abs(dx) >= abs(dy):
            return int(BossActionID.MOVE_RIGHT) if dx > 0 else int(BossActionID.MOVE_LEFT)
        return int(BossActionID.MOVE_DOWN) if dy > 0 else int(BossActionID.MOVE_UP)

    # 3. 어그로 1위가 나면 탱커 쪽으로 이동해서 어그로 전환 유도
    top_uid = env.boss.top_aggro_uid()
    if top_uid == u.uid:
        # 탱커(role=1) 찾기
        tank = next((x for x in env.units.values() if int(x.role) == 1 and x.alive), None)
        if tank is not None:
            dx = tank.x - u.x; dy = tank.y - u.y
            if _m.hypot(dx, dy) > 1.5:  # 탱커에 너무 붙진 않게
                if abs(dx) >= abs(dy):
                    return int(BossActionID.MOVE_RIGHT) if dx > 0 else int(BossActionID.MOVE_LEFT)
                return int(BossActionID.MOVE_DOWN) if dy > 0 else int(BossActionID.MOVE_UP)

    # 4. 사거리 내 공격
    if env._boss_dist(u.x, u.y) <= u.attack_range:
        if u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) <= 0:
            return int(BossActionID.ATTACK_SKILL)
        return int(BossActionID.ATTACK_BASIC)

    # 5. 탱커가 어그로 잡았을 때만 접근 (무모한 돌진 방지)
    if top_uid is not None and top_uid != u.uid:
        dx = env.boss.x - u.x; dy = env.boss.y - u.y
        if abs(dx) >= abs(dy):
            return int(BossActionID.MOVE_RIGHT) if dx > 0 else int(BossActionID.MOVE_LEFT)
        return int(BossActionID.MOVE_DOWN) if dy > 0 else int(BossActionID.MOVE_UP)

    # 탱커가 어그로 못 잡았고 거리 멀면 대기 (혼자 죽으러 안 감)
    return int(BossActionID.STAY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--run-name", type=str, default="v1",
                        help="실험 이름. boss/<run-name>/ 밑에 모델·로그·그래프 저장")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="(고급) 저장 루트 직접 지정. 기본은 boss/<run-name>")
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="체크포인트 경로 (예: boss/v7/models/ckpt_1500.pt)")
    parser.add_argument("--ent-coef", type=float, default=0.08,
                        help="PPO 엔트로피 계수 (기본 0.08, plateau/regression 시 상향)")
    args = parser.parse_args()

    # 기본 저장 위치: boss/<run-name>/
    if args.save_dir is None:
        args.save_dir = os.path.join("boss", args.run_name)
    models_dir = os.path.join(args.save_dir, "models")
    plots_dir = os.path.join(args.save_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    device = torch.device(args.device)

    cfg = BossConfig()
    env = BossRaidEnv(cfg)

    # ── 역할별 독립 네트워크 (Tank, Healer, Support 각각 별도 학습) ──
    obs_size = cfg.obs_size
    npc_slots = [i for i, r in enumerate(cfg.party_roles) if r != PartyRole.DEALER]
    uid_to_role = {i: cfg.party_roles[i] for i in npc_slots}   # {1:TANK, 2:HEALER, 3:SUPPORT}
    role_names = {PartyRole.TANK: "tank", PartyRole.HEALER: "healer", PartyRole.SUPPORT: "support"}

    nets = {
        role: ActorCritic(obs_size=obs_size, action_size=cfg.num_actions, hidden_size=256).to(device)
        for role in uid_to_role.values()
    }
    optims = {role: torch.optim.Adam(nets[role].parameters(), lr=args.lr) for role in nets}

    # 체크포인트 이어받기 (3개 네트워크 모두)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        if "nets" in ckpt:  # 새 포맷 (역할별)
            for role, state in ckpt["nets"].items():
                r_enum = PartyRole[role.upper()] if isinstance(role, str) else role
                if r_enum in nets:
                    nets[r_enum].load_state_dict(state)
            print(f"[RESUME] loaded role-separated nets from {args.resume} (trained {ckpt.get('episode', '?')} eps)")
        elif "net" in ckpt:  # 구 포맷 (shared) → 모든 역할에 같은 가중치로 초기화
            for role in nets:
                nets[role].load_state_dict(ckpt["net"])
            print(f"[RESUME] loaded shared net into all 3 role-nets (fallback): {args.resume}")
    elif args.resume:
        print(f"[WARN] resume 경로 없음: {args.resume} — 처음부터 학습")

    # ── 역할별 롤아웃 버퍼 ──
    buffers = {role: {"obs": [], "act": [], "logp": [], "rew": [], "val": [], "done": []}
               for role in nets}

    # ── CSV 로깅 ──
    import csv, json
    log_path = os.path.join(args.save_dir, "training_log.csv")
    # 설정 메타 저장 (어떤 하이퍼파라미터로 돌렸는지 추후 확인용)
    meta = {
        "run_name": args.run_name,
        "episodes_planned": args.episodes,
        "lr": args.lr,
        "device": str(device),
        "obs_size": cfg.obs_size,
        "num_actions": cfg.num_actions,
        "boss_max_hp": cfg.boss_max_hp,
        "map_size": (cfg.map_width, cfg.map_height),
        "rewards": {k: getattr(cfg, k) for k in dir(cfg) if k.startswith("rw_")},
    }
    with open(os.path.join(args.save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "episode", "steps", "result",
        "boss_hp_remaining", "boss_hp_damaged", "boss_hp_damaged_pct", "boss_hp_ratio",
        "phase_reached", "avg_reward_all",
        "reward_dealer", "reward_tank", "reward_healer", "reward_support",
        "damage_dealer", "damage_tank", "damage_healer", "damage_support",
        "damage_total",
        "heal_done_healer", "buffs_support",
        "mechanic_success", "mechanic_fail",
        "stagger_success", "stagger_fail",
        "elapsed_sec",
    ])
    log_file.flush()

    ep_rewards = []
    ep_steps = []
    victories = 0
    start = time.time()

    # 콘솔 출력용 최근 구간 추적
    _recent_boss_dmg: list = []
    _recent_wipe_count = 0

    def _recent_wipe_count_reset():
        nonlocal _recent_wipe_count
        _recent_wipe_count = 0

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        ep_step = 0
        done = False

        # 에피소드 단위 통계 누적
        ep_reward_per_uid = {i: 0.0 for i in range(4)}
        ep_damage_per_uid = {i: 0 for i in range(4)}
        ep_heal_per_uid = {i: 0 for i in range(4)}
        ep_buffs_per_uid = {i: 0 for i in range(4)}
        ep_mech_success = 0
        ep_mech_fail = 0
        ep_stagger_success = 0
        ep_stagger_fail = 0
        max_phase = 0

        # 에피소드별 BT/RL 발동 카운트 (통계용)
        ep_bt_hits = 0
        ep_rl_hits = 0

        while not done:
            actions = {}
            actions[f"p{cfg.player_slot}"] = dealer_fsm_action(env)

            # Layer 1 BT 먼저 체크, 없으면 Layer 2 RL
            step_transitions = []
            for uid in npc_slots:
                aid = f"p{uid}"
                role = uid_to_role[uid]
                # ── Layer 1: BT ──
                bt_action = BTPolicy(env, uid).act()
                if bt_action is not None:
                    # BT 가 처리 → RL buffer 에 저장 안 함 (학습 대상 아님)
                    actions[aid] = int(bt_action)
                    ep_bt_hits += 1
                else:
                    # ── Layer 2: RL ──
                    o = torch.as_tensor(obs[aid], dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        action, log_prob, value = nets[role].get_action(o)
                    a = int(action.item())
                    actions[aid] = a
                    step_transitions.append((uid, role, obs[aid], a, log_prob.item(), value.item()))
                    ep_rl_hits += 1

            obs_next, rewards, dones, infos = env.step(actions)

            # RL 이 선택한 스텝만 buffer 에 저장
            for uid, role, o, a, lp, v in step_transitions:
                aid = f"p{uid}"
                b = buffers[role]
                b["obs"].append(o)
                b["act"].append(a)
                b["logp"].append(lp)
                b["val"].append(v)
                b["rew"].append(rewards[aid])
                b["done"].append(dones[aid])

            # 에피소드 통계 수집
            for uid in range(4):
                ep_reward_per_uid[uid] += rewards.get(f"p{uid}", 0.0)
                for e in env.step_events.get(uid, []):
                    t = e.get("type")
                    if t == "damage":
                        ep_damage_per_uid[uid] += e.get("amount", 0)
                    elif t == "heal":
                        ep_heal_per_uid[uid] += e.get("amount", 0)
                    elif t == "buff":
                        ep_buffs_per_uid[uid] += 1
                    elif t == "mechanic_success":
                        ep_mech_success += 1
                    elif t == "mechanic_fail":
                        ep_mech_fail += 1
                    elif t == "stagger_success":
                        ep_stagger_success += 1
                    elif t == "stagger_fail":
                        ep_stagger_fail += 1
            max_phase = max(max_phase, int(env.boss.phase))

            ep_step += 1
            done = all(dones.values())
            obs = obs_next

        # NPC 평균 보상 (학습 대상만)
        avg_rew_npc = sum(ep_reward_per_uid[i] for i in npc_slots) / len(npc_slots)
        ep_rewards.append(avg_rew_npc)
        ep_steps.append(ep_step)
        if env.victory:
            victories += 1

        # 중복 계산 방지: mech_success/fail은 유닛 수(4)만큼 발생하므로 나눔
        ep_mech_success //= 4
        ep_mech_fail //= 4
        ep_stagger_success //= 4
        ep_stagger_fail //= 4

        # 보스 데미지 계산
        boss_hp_damaged = cfg.boss_max_hp - env.boss.hp
        boss_hp_damaged_pct = (boss_hp_damaged / cfg.boss_max_hp) * 100
        damage_total = sum(ep_damage_per_uid.values())

        # 콘솔 요약용 추적
        _recent_boss_dmg.append(boss_hp_damaged)
        if len(_recent_boss_dmg) > args.log_interval:
            _recent_boss_dmg.pop(0)
        if env.wipe:
            _recent_wipe_count += 1

        # CSV 기록
        result = "VICTORY" if env.victory else ("WIPE" if env.wipe else "TIMEOUT")
        elapsed = time.time() - start
        log_writer.writerow([
            episode, ep_step, result,
            env.boss.hp, boss_hp_damaged, round(boss_hp_damaged_pct, 2),
            round(env.boss.hp / cfg.boss_max_hp, 4),
            max_phase, round(avg_rew_npc, 3),
            round(ep_reward_per_uid[0], 3),
            round(ep_reward_per_uid[1], 3),
            round(ep_reward_per_uid[2], 3),
            round(ep_reward_per_uid[3], 3),
            ep_damage_per_uid[0], ep_damage_per_uid[1],
            ep_damage_per_uid[2], ep_damage_per_uid[3],
            damage_total,
            ep_heal_per_uid[2], ep_buffs_per_uid[3],
            ep_mech_success, ep_mech_fail,
            ep_stagger_success, ep_stagger_fail,
            round(elapsed, 1),
        ])
        log_file.flush()   # 매 에피소드 flush (버퍼링 방지)

        # PPO 업데이트 — 역할별 네트워크 각각 업데이트 (버퍼는 에피소드 끝마다 done=True로 마감)
        for role, b in buffers.items():
            if len(b["obs"]) >= 1024:
                _ppo_update(nets[role], optims[role],
                            b["obs"], b["act"], b["logp"], b["rew"], b["val"], b["done"],
                            device, ent_coef=args.ent_coef)
                for k in b: b[k].clear()

        if episode % args.log_interval == 0:
            # 최근 구간 통계
            recent_start = max(0, episode - args.log_interval)
            avg_r = np.mean(ep_rewards[-args.log_interval:])
            avg_s = np.mean(ep_steps[-args.log_interval:])
            wr = victories / episode

            # 최근 구간 보스 데미지 요약 (CSV에 쓴 거 다시 쓰지 않고 내부 추적)
            recent_damages = _recent_boss_dmg[-args.log_interval:] if _recent_boss_dmg else [0]
            avg_dmg = sum(recent_damages) / max(1, len(recent_damages))
            avg_dmg_pct = avg_dmg / cfg.boss_max_hp * 100
            max_dmg = max(recent_damages)
            max_dmg_pct = max_dmg / cfg.boss_max_hp * 100
            wipe_rate = _recent_wipe_count * 100.0 / max(1, min(args.log_interval, episode))
            _recent_wipe_count_reset()

            print(f"[EP {episode:5d}] R={avg_r:+.2f}  steps={avg_s:.1f}  "
                  f"boss_dmg: avg={avg_dmg:.0f}({avg_dmg_pct:.1f}%) max={max_dmg:.0f}({max_dmg_pct:.1f}%)  "
                  f"wipe={wipe_rate:.0f}%  vic={wr*100:.1f}%  elapsed={elapsed:.0f}s",
                  flush=True)

        if episode % args.save_interval == 0:
            path = os.path.join(models_dir, f"ckpt_{episode}.pt")
            torch.save({
                "nets": {role_names[r]: nets[r].state_dict() for r in nets},
                "episode": episode,
            }, path)
            print(f"  ↳ saved {path}", flush=True)

    log_file.close()
    torch.save({
        "nets": {role_names[r]: nets[r].state_dict() for r in nets},
        "episode": args.episodes,
    }, os.path.join(models_dir, "final.pt"))
    print(f"Training done.")
    print(f"  Models:  {models_dir}")
    print(f"  Log:     {log_path}")
    print(f"  Plots:   {plots_dir}  (run plot_boss_training.py to generate)")


def _ppo_update(net, optim, obs, act, old_logp, rew, val, done, device,
                gamma=0.99, lam=0.95, clip=0.2, epochs=4, batch=64,
                ent_coef=0.08):  # CLI 기본값과 일치
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
            loss = policy_loss + 0.5 * value_loss - ent_coef * ent
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optim.step()


if __name__ == "__main__":
    main()
