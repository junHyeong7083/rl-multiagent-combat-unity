"""Unity ↔ Python 실시간 브릿지 (보스 레이드)

- UDP (5005): 매 턴 게임 상태를 Unity로 전송
- TCP (5006): Unity에서 플레이어(딜러) 입력 수신
- NPC 3명은 학습된 PPO 모델 또는 FSM으로 구동
- 전투 속도: 기본 0.5초/턴 (플레이어 반응 가능한 속도)

사용법:
    python boss_streamer.py --mode rl --ckpt models_boss_v1/final.pt
    python boss_streamer.py --mode fsm
"""
import argparse
import json
import socket
import threading
import time
from queue import Queue, Empty

import numpy as np
import torch

from src.boss import BossConfig, BossRaidEnv, FSMNpcPolicy, PartyRole, BTPolicy, HybridPolicy
from src.boss.config import BossActionID
from src.agent import ActorCritic


UDP_PORT = 5005
TCP_PORT = 5006
TURN_INTERVAL = 0.3  # 초 / 턴 — 너무 짧으면 FSM/RL 반응 + Unity 보간이 따라가지 못함


ROLE_NAMES = {0: "DEAL", 1: "TANK", 2: "HEAL", 3: "SUPP"}
ACTION_NAMES = {
    0: "STAY", 1: "UP  ", 2: "DOWN", 3: "LEFT", 4: "RGHT",
    5: "ATK ", 6: "SKIL", 7: "TAUN", 8: "GURD",
    9: "HEAL", 10: "CLNS", 11: "BFAT", 12: "BFSH",
}
PATTERN_NAMES = {0: "Slash", 1: "Charge", 2: "Eruption", 3: "TailSwipe",
                 4: "Mark", 5: "Stagger", 6: "CrossInferno", 7: "CursedChain",
                 8: "SealBreak"}


def _log_turn(env, actions, level):
    """매 턴 콘솔 로그."""
    b = env.boss
    hp_pct = int(100 * b.hp / env.config.boss_max_hp)
    phase = f"P{int(b.phase)+1}"
    status = []
    if b.invuln_turns > 0: status.append(f"INVULN{b.invuln_turns}")
    if b.grog_turns > 0: status.append(f"GROG{b.grog_turns}")
    if b.stagger_active: status.append(f"STAGR({int(b.stagger_gauge)})")
    status_s = " ".join(status) if status else "---"

    # 보스 라인
    print(f"[T{env.current_step:3d}] BOSS {phase} HP={b.hp:4d}({hp_pct:3d}%) "
          f"pos=({b.x:4.1f},{b.y:4.1f}) {status_s}")

    # 활성 텔레그래프
    for tg in b.telegraphs:
        pname = PATTERN_NAMES.get(int(tg.pattern_id), "?")
        targets = ",".join(str(x) for x in tg.target_unit_ids) if tg.target_unit_ids else "-"
        print(f"        ⚠ {pname} in {tg.turns_remaining}t (targets: {targets})")

    if level == "full":
        # 유닛별 상세
        for uid in sorted(env.units.keys()):
            u = env.units[uid]
            if not u.alive:
                print(f"        [{ROLE_NAMES.get(int(u.role),'?')}{uid}] ✗ DEAD")
                continue
            act_name = ACTION_NAMES.get(actions.get(f"p{uid}", 0), "?")
            aggro = b.aggro.get(uid, 0.0)
            dbg = []
            if u.buff_atk > 0: dbg.append("atk")
            if u.buff_shield > 0: dbg.append("shd")
            if u.marked_turns > 0: dbg.append("MARK")
            if u.chain_turns > 0: dbg.append(f"chain→{u.chained_with}")
            dbg_s = ",".join(dbg) if dbg else ""
            print(f"        [{ROLE_NAMES.get(int(u.role),'?')}{uid}] HP={u.hp:3d}/{u.max_hp:3d} "
                  f"pos=({u.x:4.1f},{u.y:4.1f}) act={act_name} aggro={aggro:5.1f} {dbg_s}")

        # 이번 턴 이벤트
        for uid, evs in env.step_events.items():
            for e in evs:
                t = e.get("type")
                if t == "damage":
                    print(f"        → [{uid}] deals {e.get('amount',0)} dmg "
                          f"({'skill' if e.get('skill') else 'basic'})")
                elif t == "heal":
                    print(f"        → [{uid}] heals [{e.get('target')}] +{e.get('amount',0)}")
                elif t == "damage_taken":
                    print(f"        ✖ [{uid}] takes {e.get('amount',0)} dmg")
                elif t == "death":
                    print(f"        💀 [{uid}] DIED")
                elif t == "taunt":
                    print(f"        [{uid}] TAUNT")
                elif t == "mechanic_success":
                    print(f"        ✓ mechanic SUCCESS ({PATTERN_NAMES.get(e.get('pattern',0),'?')})")
                elif t == "mechanic_fail":
                    print(f"        ✗ mechanic FAIL ({PATTERN_NAMES.get(e.get('pattern',0),'?')})")
                elif t == "stagger_success":
                    print(f"        ★ STAGGER SUCCESS — boss groggy!")
                elif t == "stagger_fail":
                    print(f"        ☠ STAGGER FAIL — party wipe damage!")
                elif t == "phase_clear":
                    print(f"        ▶ PHASE CLEAR")


# ─────────────────── TCP 입력 수신 ───────────────────

class PlayerInputReceiver:
    """Unity → Python TCP 수신 스레드. 최신 입력 1개만 보관."""

    def __init__(self, port: int = TCP_PORT):
        self.port = port
        self.latest_action = int(BossActionID.STAY)
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True

    def get_action(self) -> int:
        with self._lock:
            a = self.latest_action
            # 이동·공격 외에는 1회 소비로 STAY 초기화
            self.latest_action = int(BossActionID.STAY)
            return a

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", self.port))
        sock.listen(1)
        sock.settimeout(1.0)
        print(f"[TCP] listening on {self.port}")
        while not self._stop:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue
            conn.settimeout(0.5)
            with conn:
                buf = b""
                while not self._stop:
                    try:
                        data = conn.recv(1024)
                    except socket.timeout:
                        continue
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            msg = json.loads(line.decode("utf-8"))
                            with self._lock:
                                self.latest_action = int(msg.get("action", 0))
                        except Exception:
                            pass


# ─────────────────── UDP 송신 ───────────────────

class StateBroadcaster:
    def __init__(self, port: int = UDP_PORT):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ("127.0.0.1", port)

    def send(self, snapshot: dict):
        try:
            data = json.dumps(snapshot).encode("utf-8")
            self.sock.sendto(data, self.addr)
        except Exception as e:
            print(f"[UDP] send error: {e}")


# ─────────────────── NPC 정책 ───────────────────

class RLNpcPolicy:
    def __init__(self, net: ActorCritic, env: BossRaidEnv, uid: int, device):
        self.net = net
        self.env = env
        self.uid = uid
        self.device = device

    def act(self) -> int:
        obs = self.env._observe(self.uid)
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.net.get_action(o, deterministic=True)
        return int(action.item())


# ─────────────────── 메인 루프 ───────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["rl", "fsm", "hybrid"], default="hybrid",
                        help="NPC 정책: rl(순수 RL) / fsm(비교군) / hybrid(BT+RL, v3 제안)")
    parser.add_argument("--ckpt", type=str, default="models_boss_v1/final.pt")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-player", action="store_true",
                        help="플레이어 없이 관전 모드 (딜러도 FSM)")
    parser.add_argument("--log-level", choices=["off", "brief", "full"], default="brief",
                        help="콘솔 로그 레벨: off=없음, brief=요약, full=상세")
    parser.add_argument("--user-test", action="store_true",
                        help="실전 시연 모드: 보스 고정 spawn + 인식 범위 진입 시 전투 시작 + max_steps 무시")
    parser.add_argument("--turn-based", action="store_true",
                        help="턴 기반: 플레이어 인풋 1번 = 1 스텝 진행 (Lerp 끊김 방지)")
    parser.add_argument("--turn-interval", type=float, default=TURN_INTERVAL,
                        help=f"실시간 모드 턴 간격 초 (기본 {TURN_INTERVAL}). 천천히 보려면 0.7~1.0")
    args = parser.parse_args()

    cfg = BossConfig()
    if args.user_test:
        cfg.user_test_mode = True
        print(f"[USER-TEST] boss spawn=({cfg.boss_spawn_x},{cfg.boss_spawn_y}), detection={cfg.boss_detection_range}m, max_steps 무시")
    env = BossRaidEnv(cfg)
    device = torch.device(args.device)

    # NPC 정책 생성
    npc_slots = [i for i, r in enumerate(cfg.party_roles) if r != PartyRole.DEALER]

    def _load_rl_policies():
        """v35 체크포인트 로드 → RLNpcPolicy 딕셔너리 반환."""
        ckpt = torch.load(args.ckpt, map_location=device)
        uid_to_role = {i: cfg.party_roles[i] for i in npc_slots}
        role_nets = {}
        if "nets" in ckpt:
            for role in set(uid_to_role.values()):
                n = ActorCritic(obs_size=cfg.obs_size, action_size=cfg.num_actions).to(device)
                state = ckpt["nets"].get(role.name.lower()) or list(ckpt["nets"].values())[0]
                n.load_state_dict(state); n.eval()
                role_nets[role] = n
        else:
            shared = ActorCritic(obs_size=cfg.obs_size, action_size=cfg.num_actions).to(device)
            shared.load_state_dict(ckpt["net"]); shared.eval()
            for role in set(uid_to_role.values()):
                role_nets[role] = shared
        return {uid: RLNpcPolicy(role_nets[uid_to_role[uid]], env, uid, device) for uid in npc_slots}

    if args.mode == "rl":
        npc_policies = _load_rl_policies()
    elif args.mode == "hybrid":
        rl_policies = _load_rl_policies()
        npc_policies = {uid: HybridPolicy(env, uid, rl_policies[uid]) for uid in npc_slots}
    else:  # fsm
        npc_policies = {uid: FSMNpcPolicy(env, uid) for uid in npc_slots}

    # 통신
    broadcaster = StateBroadcaster()
    receiver = None
    if not args.no_player:
        receiver = PlayerInputReceiver()
        receiver.start()

    print(f"[BOSS] mode={args.mode}, device={device}, player={'human' if receiver else 'fsm'}")
    broadcaster.send(env.get_snapshot())

    turn_interval = args.turn_interval
    if args.turn_based:
        print("[TURN-BASED] 1 인풋 = 1 스텝. STAY 입력은 무시. 키 누를 때마다 진행")
    else:
        print(f"[REALTIME] 턴 간격 = {turn_interval}초")

    def _run_step(player_action):
        """한 스텝 실행 + 스냅샷 송신 + 로그."""
        actions = {f"p{cfg.player_slot}": player_action}
        for uid, policy in npc_policies.items():
            actions[f"p{uid}"] = policy.act()
        env.step(actions)
        broadcaster.send(env.get_snapshot())
        if args.log_level != "off":
            _log_turn(env, actions, args.log_level)

    try:
        while True:
            obs = env.reset()
            broadcaster.send(env.get_snapshot())
            time.sleep(turn_interval)

            while not env.done:
                if args.turn_based and receiver:
                    # 턴 기반: STAY 가 아닌 입력이 들어올 때까지 대기
                    pa = int(BossActionID.STAY)
                    while pa == int(BossActionID.STAY) and not env.done:
                        pa = receiver.get_action()
                        if pa == int(BossActionID.STAY):
                            time.sleep(0.02)
                    if env.done: break
                    _run_step(pa)
                else:
                    # 실시간 모드 (기존)
                    if receiver:
                        pa = receiver.get_action()
                    else:
                        from train_boss import dealer_fsm_action
                        pa = dealer_fsm_action(env)
                    _run_step(pa)
                    time.sleep(turn_interval)

            result = "VICTORY" if env.victory else ("WIPE" if env.wipe else "TIMEOUT")
            print(f"\n{'='*60}\n[BOSS] episode end: {result}, steps={env.current_step}\n{'='*60}")
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[BOSS] interrupted")
    finally:
        if receiver:
            receiver.stop()


if __name__ == "__main__":
    main()
