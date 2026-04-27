"""RL vs FSM NPC 평가 스크립트.

사용법:
    # RL 모델 평가
    python eval_boss.py --mode rl --ckpt boss/v10/models/final.pt --episodes 100

    # FSM NPC 평가 (비교 baseline)
    python eval_boss.py --mode fsm --episodes 100

    # 두 개 동시 (한 번에 A/B 비교)
    python eval_boss.py --compare boss/v10/models/final.pt --episodes 100
"""
import argparse
import os
import numpy as np
import torch
from collections import defaultdict

from src.boss import BossConfig, BossRaidEnv, FSMNpcPolicy, PartyRole
from src.boss.config import BossActionID
from src.agent import ActorCritic
from train_boss import dealer_fsm_action


class RLNpcPolicy:
    def __init__(self, net, env, uid, device):
        self.net, self.env, self.uid, self.device = net, env, uid, device

    def act(self):
        obs = self.env._observe(self.uid)
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.net.get_action(o, deterministic=True)
        return int(action.item())


def load_role_nets(ckpt_path, cfg, npc_slots, device):
    """신/구 포맷 모두 지원. 역할별 네트워크 dict 반환."""
    uid_to_role = {i: cfg.party_roles[i] for i in npc_slots}
    nets = {}
    ckpt = torch.load(ckpt_path, map_location=device)
    if "nets" in ckpt:  # 신 포맷
        for role in set(uid_to_role.values()):
            n = ActorCritic(obs_size=cfg.obs_size, action_size=cfg.num_actions).to(device)
            state = ckpt["nets"].get(role.name.lower())
            if state is None:
                state = list(ckpt["nets"].values())[0]
            n.load_state_dict(state)
            n.eval()
            nets[role] = n
    else:  # 구 포맷 (shared net) — 모든 역할에 같은 가중치
        shared = ActorCritic(obs_size=cfg.obs_size, action_size=cfg.num_actions).to(device)
        shared.load_state_dict(ckpt["net"])
        shared.eval()
        for role in set(uid_to_role.values()):
            nets[role] = shared
    return nets, uid_to_role


def run_episodes(env, npc_policies, n, verbose=False):
    stats = defaultdict(list)
    for ep in range(n):
        obs = env.reset(seed=ep)  # seed 고정으로 공정 비교
        step_count = 0
        ep_damage = {i: 0 for i in range(4)}
        ep_mech_success = 0
        ep_mech_fail = 0

        while not env.done:
            actions = {f"p{env.config.player_slot}": dealer_fsm_action(env)}
            for uid, pol in npc_policies.items():
                actions[f"p{uid}"] = pol.act()
            env.step(actions)
            step_count += 1

            for uid, evs in env.step_events.items():
                for e in evs:
                    if e.get("type") == "damage":
                        ep_damage[uid] += e.get("amount", 0)
                    elif e.get("type") == "mechanic_success":
                        ep_mech_success += 1
                    elif e.get("type") == "mechanic_fail":
                        ep_mech_fail += 1

        result = "VICTORY" if env.victory else ("WIPE" if env.wipe else "TIMEOUT")
        stats["result"].append(result)
        stats["steps"].append(step_count)
        stats["boss_hp_remaining"].append(env.boss.hp)
        stats["boss_dmg_pct"].append(100 - env.boss.hp / env.config.boss_max_hp * 100)
        stats["phase_reached"].append(int(env.boss.phase))
        for uid in range(4):
            stats[f"damage_p{uid}"].append(ep_damage[uid])
        stats["mechanic_success"].append(ep_mech_success // 4)
        stats["mechanic_fail"].append(ep_mech_fail // 4)

        if verbose and (ep + 1) % 10 == 0:
            win = sum(1 for r in stats["result"] if r == "VICTORY")
            print(f"  [{ep+1}/{n}] vic={win}/{ep+1} ({win/(ep+1)*100:.1f}%)")
    return stats


def summarize(stats, label=""):
    n = len(stats["result"])
    victories = sum(1 for r in stats["result"] if r == "VICTORY")
    wipes = sum(1 for r in stats["result"] if r == "WIPE")
    timeouts = sum(1 for r in stats["result"] if r == "TIMEOUT")

    print(f"\n{'='*60}")
    print(f" {label} 평가 결과 ({n} 에피소드)")
    print(f"{'='*60}")
    print(f"  Victory:        {victories:4d} ({victories/n*100:5.1f}%)")
    print(f"  Wipe:           {wipes:4d} ({wipes/n*100:5.1f}%)")
    print(f"  Timeout:        {timeouts:4d} ({timeouts/n*100:5.1f}%)")
    print()
    print(f"  평균 스텝:       {np.mean(stats['steps']):6.1f}")
    print(f"  평균 보스 딜:    {np.mean(stats['boss_dmg_pct']):6.1f}%")
    print(f"  최대 보스 딜:    {np.max(stats['boss_dmg_pct']):6.1f}%")
    print(f"  평균 도달 페이즈:{np.mean(stats['phase_reached']):6.2f}")
    print()
    print(f"  역할별 평균 딜량:")
    print(f"    Dealer:  {np.mean(stats['damage_p0']):6.1f}")
    print(f"    Tank:    {np.mean(stats['damage_p1']):6.1f}")
    print(f"    Healer:  {np.mean(stats['damage_p2']):6.1f}")
    print(f"    Support: {np.mean(stats['damage_p3']):6.1f}")
    print()
    print(f"  기믹 성공/실패:  {np.mean(stats['mechanic_success']):.2f} / {np.mean(stats['mechanic_fail']):.2f}")

    return {
        "victory_rate": victories / n,
        "avg_dmg_pct": float(np.mean(stats['boss_dmg_pct'])),
        "avg_steps": float(np.mean(stats['steps'])),
        "avg_phase": float(np.mean(stats['phase_reached'])),
        "mech_success": float(np.mean(stats['mechanic_success'])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rl", "fsm"], default="rl")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--compare", type=str, default=None,
                    help="주어진 ckpt로 RL, 같은 env로 FSM 둘 다 평가")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = BossConfig()
    device = torch.device(args.device)
    npc_slots = [i for i, r in enumerate(cfg.party_roles) if r != PartyRole.DEALER]

    results = {}

    if args.compare:
        # 두 모드 순차 평가 (같은 seed로 공정 비교)
        # 1) RL
        print(f"\n[1/2] RL NPC 평가 ({args.compare})")
        env = BossRaidEnv(cfg)
        role_nets, uid_to_role = load_role_nets(args.compare, cfg, npc_slots, device)
        policies = {uid: RLNpcPolicy(role_nets[uid_to_role[uid]], env, uid, device) for uid in npc_slots}
        stats_rl = run_episodes(env, policies, args.episodes, verbose=True)
        results["RL"] = summarize(stats_rl, label="RL NPC")

        # 2) FSM
        print(f"\n[2/2] FSM NPC 평가 (동일 환경)")
        env = BossRaidEnv(cfg)
        policies = {uid: FSMNpcPolicy(env, uid) for uid in npc_slots}
        stats_fsm = run_episodes(env, policies, args.episodes, verbose=True)
        results["FSM"] = summarize(stats_fsm, label="FSM NPC (baseline)")

        # 비교 요약
        print(f"\n{'='*60}")
        print(f" RL vs FSM 비교 요약 ({args.episodes} 에피소드씩)")
        print(f"{'='*60}")
        print(f"  지표                RL          FSM       차이(배)")
        print(f"  {'-'*55}")
        rl, fsm = results["RL"], results["FSM"]
        ratio = lambda a, b: f"{a/max(b, 1e-9):.2f}x" if b > 0 else "∞"
        print(f"  승률             {rl['victory_rate']*100:6.1f}%   {fsm['victory_rate']*100:6.1f}%   {ratio(rl['victory_rate'], fsm['victory_rate'])}")
        print(f"  보스 딜 평균     {rl['avg_dmg_pct']:6.1f}%   {fsm['avg_dmg_pct']:6.1f}%   {ratio(rl['avg_dmg_pct'], fsm['avg_dmg_pct'])}")
        print(f"  생존 스텝        {rl['avg_steps']:6.1f}    {fsm['avg_steps']:6.1f}    {ratio(rl['avg_steps'], fsm['avg_steps'])}")
        print(f"  페이즈 도달      {rl['avg_phase']:6.2f}    {fsm['avg_phase']:6.2f}    {ratio(rl['avg_phase'], fsm['avg_phase'])}")
        print(f"  기믹 성공        {rl['mech_success']:6.2f}    {fsm['mech_success']:6.2f}    {ratio(rl['mech_success'], fsm['mech_success'])}")
        print()

    elif args.mode == "rl":
        if not args.ckpt:
            print("Error: --ckpt 필요"); return
        env = BossRaidEnv(cfg)
        role_nets, uid_to_role = load_role_nets(args.ckpt, cfg, npc_slots, device)
        policies = {uid: RLNpcPolicy(role_nets[uid_to_role[uid]], env, uid, device) for uid in npc_slots}
        stats = run_episodes(env, policies, args.episodes, verbose=True)
        summarize(stats, label=f"RL NPC ({os.path.basename(args.ckpt)})")

    else:  # fsm
        env = BossRaidEnv(cfg)
        policies = {uid: FSMNpcPolicy(env, uid) for uid in npc_slots}
        stats = run_episodes(env, policies, args.episodes, verbose=True)
        summarize(stats, label="FSM NPC (baseline)")


if __name__ == "__main__":
    main()
