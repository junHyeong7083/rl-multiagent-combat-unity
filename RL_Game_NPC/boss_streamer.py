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

from src.boss import BossConfig, BossRaidEnv, FSMNpcPolicy, PartyRole
from src.boss.config import BossActionID
from src.agent import ActorCritic


UDP_PORT = 5005
TCP_PORT = 5006
TURN_INTERVAL = 0.5  # 초 / 턴


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
    parser.add_argument("--mode", choices=["rl", "fsm"], default="rl",
                        help="NPC 정책: rl(학습된 모델) 또는 fsm(비교군)")
    parser.add_argument("--ckpt", type=str, default="models_boss_v1/final.pt")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-player", action="store_true",
                        help="플레이어 없이 관전 모드 (딜러도 FSM)")
    args = parser.parse_args()

    cfg = BossConfig()
    env = BossRaidEnv(cfg)
    device = torch.device(args.device)

    # NPC 정책 생성
    npc_slots = [i for i, r in enumerate(cfg.party_roles) if r != PartyRole.DEALER]
    if args.mode == "rl":
        net = ActorCritic(obs_size=cfg.obs_size, action_size=cfg.num_actions).to(device)
        ckpt = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(ckpt["net"])
        net.eval()
        npc_policies = {uid: RLNpcPolicy(net, env, uid, device) for uid in npc_slots}
    else:
        npc_policies = {uid: FSMNpcPolicy(env, uid) for uid in npc_slots}

    # 통신
    broadcaster = StateBroadcaster()
    receiver = None
    if not args.no_player:
        receiver = PlayerInputReceiver()
        receiver.start()

    print(f"[BOSS] mode={args.mode}, device={device}, player={'human' if receiver else 'fsm'}")
    broadcaster.send(env.get_snapshot())

    try:
        while True:
            obs = env.reset()
            broadcaster.send(env.get_snapshot())
            time.sleep(TURN_INTERVAL)

            while not env.done:
                actions = {}
                # 플레이어(딜러) 입력
                if receiver:
                    actions[f"p{cfg.player_slot}"] = receiver.get_action()
                else:
                    # FSM 딜러 (학습용 더미)
                    from train_boss import dealer_fsm_action
                    actions[f"p{cfg.player_slot}"] = dealer_fsm_action(env)

                # NPC 행동
                for uid, policy in npc_policies.items():
                    actions[f"p{uid}"] = policy.act()

                env.step(actions)
                broadcaster.send(env.get_snapshot())
                time.sleep(TURN_INTERVAL)

            # 종료
            result = "VICTORY" if env.victory else ("WIPE" if env.wipe else "TIMEOUT")
            print(f"[BOSS] episode end: {result}, steps={env.current_step}")
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[BOSS] interrupted")
    finally:
        if receiver:
            receiver.stop()


if __name__ == "__main__":
    main()
