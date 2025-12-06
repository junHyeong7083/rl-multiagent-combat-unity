"""
Player Mode Streamer - 플레이어가 A팀 탱커를 조종하고 나머지 NPC는 AI가 협동
Unity로 UDP 전송, Unity에서 TCP로 플레이어 입력 수신

v11 협동 모델 사용:
- A팀 탱커 (index 0): 플레이어가 직접 조종
- A팀 NPC (index 1~4): v11 협동 모델 (235차원) - 플레이어 따라가기 학습됨
- B팀: v11 일반 모델 (229차원) - 적 AI

obs 차원:
- env obs: 229차원
- v11 coop NPC model: 235차원 (229 + tank_info 6)
- v11 일반 model: 229차원
"""
import os
import sys
import json
import socket
import time
import argparse
import threading
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.config import EnvConfig, RoleType, ActionType
from src.env import MultiAgentBattleEnv
from src.agent import PPOAgent


# 탱커 인덱스 (v7과 동일)
PLAYER_IDX = 0


def get_tank_info(env, npc_id, tank_id):
    """NPC observation에 추가할 탱커 정보 (6차원) - train_coop_v7.py와 동일"""
    if tank_id not in env.units:
        return np.zeros(6, dtype=np.float32)

    npc = env.units[npc_id]
    tank = env.units[tank_id]

    return np.array([
        (tank.x - npc.x) / env.config.map_width,   # 상대 x
        (tank.y - npc.y) / env.config.map_height,  # 상대 y
        npc.distance_to(tank) / 20.0,              # 거리 정규화
        float(tank.is_alive),                      # 탱커 생존
        tank.hp / tank.max_hp if tank.max_hp > 0 else 0,  # HP 비율
        1.0  # 탱커 존재 플래그
    ], dtype=np.float32)


class PlayerInputReceiver:
    """Unity에서 플레이어 입력을 TCP로 수신"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5006):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.connected = False

        # 최신 플레이어 입력
        self.player_role = 0  # 선택된 역할 (0~4)
        self.player_action = 0  # 현재 액션 (0=STAY)
        self.action_queue = deque(maxlen=10)

        self._lock = threading.Lock()

    def start(self):
        """서버 시작"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.settimeout(1.0)  # accept timeout
        self.running = True

        self.receive_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.receive_thread.start()

        print(f"[PlayerInputReceiver] Listening on {self.host}:{self.port}")

    def _accept_loop(self):
        """클라이언트 연결 대기 및 수신"""
        while self.running:
            try:
                # 클라이언트 연결 대기
                if self.client_socket is None:
                    try:
                        self.client_socket, addr = self.server_socket.accept()
                        self.client_socket.settimeout(0.1)
                        self.connected = True
                        print(f"[PlayerInputReceiver] Unity connected from {addr}")
                    except socket.timeout:
                        continue

                # 데이터 수신
                try:
                    data = self.client_socket.recv(1024)
                    if not data:
                        self._disconnect()
                        continue

                    # 줄바꿈으로 구분된 JSON 파싱
                    messages = data.decode('utf-8').strip().split('\n')
                    for msg in messages:
                        if msg:
                            self._process_message(msg)

                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[PlayerInputReceiver] Receive error: {e}")
                    self._disconnect()

            except Exception as e:
                if self.running:
                    print(f"[PlayerInputReceiver] Error: {e}")

    def _process_message(self, msg: str):
        """수신된 메시지 처리"""
        try:
            data = json.loads(msg)
            msg_type = data.get('type', '')

            with self._lock:
                if msg_type == 'role':
                    self.player_role = data.get('role', 0)
                    print(f"[PlayerInputReceiver] Player selected role: {RoleType(self.player_role).name}")

                elif msg_type == 'action':
                    action = data.get('action', 0)
                    self.player_action = action
                    self.action_queue.append(action)

        except json.JSONDecodeError as e:
            print(f"[PlayerInputReceiver] JSON error: {e}")

    def _disconnect(self):
        """클라이언트 연결 해제"""
        self.connected = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        print("[PlayerInputReceiver] Unity disconnected")

    def get_player_action(self) -> int:
        """현재 플레이어 액션 반환"""
        with self._lock:
            return self.player_action

    def get_player_role(self) -> int:
        """선택된 플레이어 역할 반환"""
        with self._lock:
            return self.player_role

    def is_connected(self) -> bool:
        """Unity 연결 상태"""
        return self.connected

    def stop(self):
        """서버 종료"""
        self.running = False
        self._disconnect()
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass


class UnityStreamer:
    """게임 상태를 Unity로 UDP 전송"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[UnityStreamer] Ready to send to {host}:{port}")

    def send_frame(self, env: MultiAgentBattleEnv, step: int, player_idx: int = -1):
        """현재 환경 상태를 Unity에 전송"""
        frame = self._build_frame(env, step, player_idx)
        data = json.dumps(frame, ensure_ascii=False)
        self.sock.sendto(data.encode('utf-8'), (self.host, self.port))

    def _build_frame(self, env: MultiAgentBattleEnv, step: int, player_idx: int = -1) -> dict:
        """FrameDTO 형식으로 데이터 구성"""

        team_a = []
        team_b = []

        for i, agent_id in enumerate(env.team_a):
            unit = env.units[agent_id]
            team_a.append({
                "x": unit.x,
                "y": unit.y,
                "hp": unit.hp,
                "maxHp": unit.max_hp,
                "mp": unit.mp,
                "maxMp": unit.max_mp,
                "role": int(unit.role),
                "alive": unit.is_alive,
                "isPlayer": i == player_idx  # 플레이어 표시
            })

        for agent_id in env.team_b:
            unit = env.units[agent_id]
            team_b.append({
                "x": unit.x,
                "y": unit.y,
                "hp": unit.hp,
                "maxHp": unit.max_hp,
                "mp": unit.mp,
                "maxMp": unit.max_mp,
                "role": int(unit.role),
                "alive": unit.is_alive,
                "isPlayer": False
            })

        tiles = env.game_map.tiles.flatten().tolist()

        winner = ""
        if env.done:
            if env.winner == 0:
                winner = "A"
            elif env.winner == 1:
                winner = "B"
            else:
                winner = "draw"

        frame = {
            "step": step,
            "mapWidth": env.config.map_width,
            "mapHeight": env.config.map_height,
            "tiles": tiles,
            "teamA": team_a,
            "teamB": team_b,
            "done": env.done,
            "winner": winner,
            "playerIdx": player_idx
        }

        return frame

    def close(self):
        self.sock.close()


def find_player_unit_index(env: MultiAgentBattleEnv, player_role: int) -> int:
    """
    플레이어가 선택한 역할에 해당하는 유닛 인덱스 찾기
    team_composition에서 해당 역할의 인덱스 반환
    """
    for i, role in enumerate(env.config.team_composition):
        if role == player_role:
            return i
    return 0  # 기본값


def play_player_mode(env: MultiAgentBattleEnv, streamer: UnityStreamer,
                     input_receiver: PlayerInputReceiver, agent_npc: PPOAgent,
                     agent_tank: PPOAgent, agent_b: PPOAgent,
                     num_episodes: int = 10, delay: float = 0.1,
                     obs_size_npc: int = None, obs_size_tank: int = None,
                     obs_size_b: int = None):
    """플레이어 모드로 게임 플레이 (A팀: 협동, B팀: 적)

    v9 호환: 각 모델에 맞는 obs_size 자동 조정
    - obs_size_npc: NPC 모델 (v7/v9: 235)
    - obs_size_tank: 탱크 모델 (v11: 229 또는 이전)
    - obs_size_b: B팀 모델 (v11: 229 또는 이전)
    """

    print("\n" + "=" * 50)
    print("Player Mode - Waiting for Unity connection...")
    print("=" * 50)

    # Unity 연결 대기
    wait_count = 0
    while not input_receiver.is_connected():
        time.sleep(0.5)
        wait_count += 1
        if wait_count % 10 == 0:
            print(f"Waiting for Unity... ({wait_count // 2}s)")
        if wait_count > 120:  # 60초 타임아웃
            print("Timeout waiting for Unity connection")
            return

    print("Unity connected! Starting game...")
    time.sleep(1)  # Unity가 준비될 시간

    # 플레이어 역할 확인
    player_role = input_receiver.get_player_role()
    player_idx = find_player_unit_index(env, player_role)
    player_agent_id = f"team_a_{player_idx}"

    # 탱커와 NPC 구분 (v7 호환)
    tank_id = f"team_a_{PLAYER_IDX}"
    npc_ids = [aid for aid in env.team_a if aid != tank_id]

    print(f"Player controls: {RoleType(player_role).name} (index {player_idx})")
    print(f"Tank: {tank_id}, NPCs: {npc_ids}")

    for ep in range(num_episodes):
        print(f"\n=== Episode {ep + 1}/{num_episodes} ===")

        obs = env.reset()

        # 플레이어 역할 재확인 (에피소드마다 바뀔 수 있음)
        player_role = input_receiver.get_player_role()
        player_idx = find_player_unit_index(env, player_role)
        player_agent_id = f"team_a_{player_idx}"

        done = False
        step = 0

        # 초기 상태 전송
        streamer.send_frame(env, step, player_idx)
        time.sleep(delay)

        while not done:
            actions = {}

            # === 탱커 행동 (v11 모델) ===
            # 탱크 모델의 obs_size와 env obs_size 맞추기
            tank_ob = obs[tank_id]
            env_obs_len = len(tank_ob)

            if obs_size_tank and obs_size_tank < env_obs_len:
                # v11 모델이 더 작은 obs 기대 → 잘라냄
                tank_ob = tank_ob[:obs_size_tank]
            elif obs_size_tank and obs_size_tank > env_obs_len:
                # 더 큰 obs 기대 → 0으로 패딩
                padding = np.zeros(obs_size_tank - env_obs_len, dtype=np.float32)
                tank_ob = np.concatenate([tank_ob, padding])

            obs_tank = {tank_id: tank_ob}
            actions_tank, _, _ = agent_tank.get_actions(obs_tank, deterministic=False)
            actions.update(actions_tank)

            # === A팀 NPC 행동 (v7 모델) ===
            # 디버그: 첫 스텝에서 obs 마지막 6차원 출력 (tank_info 확인)
            if step == 0 and ep == 0:
                for aid in npc_ids[:1]:  # 첫 NPC만
                    npc_obs = obs[aid]
                    print(f"[DEBUG] {aid} obs shape: {npc_obs.shape}")
                    print(f"[DEBUG] Last 6 dims (should be tank_info): {npc_obs[-6:]}")
                    tank = env.units[tank_id]
                    npc = env.units[aid]
                    print(f"[DEBUG] Tank pos: ({tank.x}, {tank.y}), NPC pos: ({npc.x}, {npc.y})")
                    print(f"[DEBUG] Distance to tank: {npc.distance_to(tank):.2f}")

            # NPC obs에 tank_info 추가 (학습 시와 동일하게 실제 탱크 정보!)
            # obs_size_npc가 env보다 크면 (235 vs 229) tank_info 6차원 추가
            obs_npc = {}
            for aid in npc_ids:
                npc_ob = obs[aid]
                env_obs_size = len(npc_ob)

                if obs_size_npc and obs_size_npc > env_obs_size:
                    # 모델이 더 큰 obs 기대 → tank_info 추가
                    tank_info = get_tank_info(env, aid, tank_id)
                    npc_ob = np.concatenate([npc_ob, tank_info])

                    # 디버그: 차원 확인
                    if step == 0 and aid == npc_ids[0]:
                        print(f"[DEBUG] NPC obs: env={env_obs_size} + tank_info=6 → {len(npc_ob)}")

                obs_npc[aid] = npc_ob
            actions_npc, _, _ = agent_npc.get_actions(obs_npc, deterministic=False)
            actions.update(actions_npc)

            # === B팀 행동 (적 모델 - v11) ===
            # v11 모델의 obs_size와 env obs_size 맞추기
            obs_b = {}
            for aid in env.team_b:
                b_ob = obs[aid]
                env_obs_len = len(b_ob)

                if obs_size_b and obs_size_b < env_obs_len:
                    # v11 모델이 더 작은 obs 기대 → 잘라냄
                    b_ob = b_ob[:obs_size_b]
                    if step == 0 and aid == env.team_b[0]:
                        print(f"[DEBUG] B팀 obs: env={env_obs_len} → trimmed to {len(b_ob)}")
                elif obs_size_b and obs_size_b > env_obs_len:
                    # v11 모델이 더 큰 obs 기대 → 0으로 패딩
                    padding = np.zeros(obs_size_b - env_obs_len, dtype=np.float32)
                    b_ob = np.concatenate([b_ob, padding])
                    if step == 0 and aid == env.team_b[0]:
                        print(f"[DEBUG] B팀 obs: env={env_obs_len} → padded to {len(b_ob)}")

                obs_b[aid] = b_ob
            actions_b, _, _ = agent_b.get_actions(obs_b, deterministic=False)
            actions.update(actions_b)

            # 디버그: AI 액션 및 위치 출력 (처음 20스텝)
            if step < 20:
                action_names = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                               5: "ATK_NEAR", 6: "ATK_LOW", 7: "AOE", 8: "HEAL",
                               9: "TAUNT", 10: "PIERCE", 11: "BUFF"}

                # 탱커와 NPC 위치 출력
                tank = env.units[tank_id]
                print(f"\n[Step {step}] Tank({tank.x},{tank.y}) | ", end="")
                for aid in npc_ids:
                    npc = env.units[aid]
                    act = actions.get(aid, -1)
                    print(f"{aid[-1]}:({npc.x},{npc.y})->{action_names.get(act, '?')}, ", end="")
                print()

            # 플레이어 유닛이 살아있으면 플레이어 입력으로 대체
            player_unit = env.units[player_agent_id]
            if player_unit.is_alive:
                player_action = input_receiver.get_player_action()
                actions[player_agent_id] = player_action

            # 환경 스텝
            obs, rewards, dones, truncated, infos = env.step(actions)
            done = all(dones.values())
            step += 1

            # 스텝 후 위치 확인 (처음 20스텝)
            if step <= 20:
                tank = env.units[tank_id]
                print(f"  → After step: Tank({tank.x},{tank.y}) | ", end="")
                for aid in npc_ids:
                    npc = env.units[aid]
                    print(f"{aid[-1]}:({npc.x},{npc.y}), ", end="")
                print()

            # Unity로 전송
            streamer.send_frame(env, step, player_idx)
            time.sleep(delay)

            # 연결 끊기면 중단
            if not input_receiver.is_connected():
                print("Unity disconnected, stopping...")
                return

            if done:
                if env.winner == 0:
                    result = "Team A Wins! (Your team)"
                elif env.winner == 1:
                    result = "Team B Wins!"
                else:
                    result = "Draw!"
                print(f"Game Over! {result} (Steps: {step})")
                time.sleep(3)  # 결과 확인 시간


def get_model_obs_size(model_path):
    """모델 파일에서 obs_size 자동 감지"""
    if not os.path.exists(model_path):
        return None
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint['network']['shared.0.weight'].shape[1]
    except Exception as e:
        print(f"[Warning] Could not read obs_size from {model_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Player Mode Streamer (v9 호환)')
    parser.add_argument('--udp-host', type=str, default='127.0.0.1', help='Unity UDP host')
    parser.add_argument('--udp-port', type=int, default=5005, help='Unity UDP port')
    parser.add_argument('--tcp-host', type=str, default='127.0.0.1', help='Player input TCP host')
    parser.add_argument('--tcp-port', type=int, default=5006, help='Player input TCP port')
    parser.add_argument('--model', type=str, default='models_coop_v11/model_npc_latest.pt',
                        help='A Team NPC model path (v11 coop: 235 dims)')
    parser.add_argument('--tank-model', type=str, default='models_v11_10k_episodes/model_final.pt',
                        help='Tank model path (v11 모델, 229 dims) - 플레이어가 조종 시 무시됨')
    parser.add_argument('--model-b', type=str, default='models_v11_10k_episodes/model_final.pt',
                        help='Team B model path (enemy team, 229 dims)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--delay', type=float, default=0.15, help='Delay between frames')
    # 스탯 배율 인자
    parser.add_argument('--stat-a', type=float, default=1.0, help='Team A stat multiplier (0.7 = 70%)')
    parser.add_argument('--stat-b', type=float, default=1.0, help='Team B stat multiplier (1.0 = 100%)')

    args = parser.parse_args()

    # 환경 생성 (스탯 배율 적용)
    env_config = EnvConfig()
    env_config.team_a_stat_multiplier = args.stat_a
    env_config.team_b_stat_multiplier = args.stat_b
    env_config.player_idx = PLAYER_IDX  # v7 협동 학습용 탱커 인덱스
    env_config.max_steps = 100000  # 실제 게임은 제한 없음 (step 정규화가 1.0 이하로 유지되도록)
    env = MultiAgentBattleEnv(env_config)

    print(f"[Config] Team A stat: {args.stat_a*100:.0f}%, Team B stat: {args.stat_b*100:.0f}%")
    print(f"[Config] Player (Tank) index: {PLAYER_IDX}")

    # AI 에이전트 생성 및 모델 로드
    obs = env.reset()
    obs_size_env = list(obs.values())[0].shape[0]  # env가 반환하는 크기 (229)
    action_size = env.action_space_size

    print(f"\n{'='*50}")
    print(f"[Config] env obs_size: {obs_size_env}")

    # === NPC 모델 obs_size 자동 감지 ===
    obs_size_npc = get_model_obs_size(args.model)
    if obs_size_npc is None:
        obs_size_npc = obs_size_env + 6  # 기본값: env(229) + tank_info(6) = 235
        print(f"[Config] NPC model not found, using default: {obs_size_npc}")
    else:
        print(f"[Config] NPC model obs_size: {obs_size_npc}")

    # 차원 차이 계산 (tank_info 추가 여부 결정)
    obs_diff = obs_size_npc - obs_size_env
    if obs_diff == 6:
        print(f"[Config] NPC 모델이 tank_info(6) 필요 → 자동 추가됨")
    elif obs_diff == 0:
        print(f"[Config] NPC 모델이 tank_info 불필요 (v11 호환)")
    else:
        print(f"[Warning] obs 차원 불일치! env={obs_size_env}, model={obs_size_npc}, diff={obs_diff}")

    # === 탱크 모델 obs_size 자동 감지 ===
    obs_size_tank = get_model_obs_size(args.tank_model)
    if obs_size_tank is None:
        obs_size_tank = obs_size_env
        print(f"[Config] Tank model not found, using env size: {obs_size_tank}")
    else:
        print(f"[Config] Tank model obs_size: {obs_size_tank}")

    # === B팀 모델 obs_size 자동 감지 ===
    obs_size_b = get_model_obs_size(args.model_b)
    if obs_size_b is None:
        obs_size_b = obs_size_env
        print(f"[Config] B Team model not found, using env size: {obs_size_b}")
    else:
        print(f"[Config] B Team model obs_size: {obs_size_b}")

    print(f"{'='*50}\n")

    # A팀 NPC 에이전트 (저장된 모델 크기에 맞춤)
    agent_npc = PPOAgent(obs_size_npc, action_size)
    if os.path.exists(args.model):
        agent_npc.load(args.model)
        print(f"Loaded A Team NPC model ({obs_size_npc} dims): {args.model}")
    else:
        print(f"Warning: A Team NPC model not found ({args.model}), using random AI")

    # A팀 탱커 에이전트 (v11 모델 또는 플레이어)
    agent_tank = PPOAgent(obs_size_tank, action_size)
    if args.tank_model and os.path.exists(args.tank_model):
        agent_tank.load(args.tank_model)
        print(f"Loaded Tank model ({obs_size_tank} dims): {args.tank_model}")
    else:
        print(f"Warning: Tank model not found ({args.tank_model}), using random AI")

    # Team B 에이전트 (적팀)
    agent_b = PPOAgent(obs_size_b, action_size)
    if args.model_b and os.path.exists(args.model_b):
        agent_b.load(args.model_b)
        print(f"Loaded Team B model ({obs_size_b} dims): {args.model_b}")
    else:
        print(f"Warning: Team B model not found ({args.model_b}), using random AI")

    # Unity 스트리머
    streamer = UnityStreamer(args.udp_host, args.udp_port)

    # 플레이어 입력 수신기
    input_receiver = PlayerInputReceiver(args.tcp_host, args.tcp_port)
    input_receiver.start()

    try:
        play_player_mode(
            env, streamer, input_receiver, agent_npc, agent_tank, agent_b,
            num_episodes=args.episodes,
            delay=args.delay,
            obs_size_npc=obs_size_npc,
            obs_size_tank=obs_size_tank,
            obs_size_b=obs_size_b
        )
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        input_receiver.stop()
        streamer.close()
        print("Player mode streamer closed")


if __name__ == '__main__':
    main()