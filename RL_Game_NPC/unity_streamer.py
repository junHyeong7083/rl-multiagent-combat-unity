"""Unity UDP Streamer - RL_Game_NPC 환경 상태를 Unity로 전송

v11 협동 모델 호환:
- A팀 협동 모델: 235차원 (229 + tank_info 6)
- B팀 일반 모델: 229차원
- tank_info 자동 추가 지원
"""
import os
import sys
import json
import socket
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.config import EnvConfig, RoleType, TileType
from src.env import MultiAgentBattleEnv
from src.agent import PPOAgent


def get_model_obs_size(model_path):
    """모델 파일에서 obs_size 자동 감지"""
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        return checkpoint['network']['shared.0.weight'].shape[1]
    except Exception as e:
        print(f"[Warning] Could not read obs_size from {model_path}: {e}")
        return None


def get_tank_info(env, npc_id, tank_id):
    """NPC observation에 추가할 탱커 정보 (6차원)"""
    if tank_id not in env.units:
        return np.zeros(6, dtype=np.float32)

    npc = env.units[npc_id]
    tank = env.units[tank_id]

    return np.array([
        (tank.x - npc.x) / env.config.map_width,
        (tank.y - npc.y) / env.config.map_height,
        npc.distance_to(tank) / 20.0,
        float(tank.is_alive),
        tank.hp / tank.max_hp if tank.max_hp > 0 else 0,
        1.0
    ], dtype=np.float32)


class UnityStreamer:
    """게임 상태를 Unity로 UDP 전송"""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[UnityStreamer] Ready to send to {host}:{port}")

    def send_frame(self, env: MultiAgentBattleEnv, step: int, actions: dict = None):
        """현재 환경 상태를 Unity에 전송"""
        frame = self._build_frame(env, step, actions)
        data = json.dumps(frame, ensure_ascii=False)
        self.sock.sendto(data.encode('utf-8'), (self.host, self.port))

    def _build_frame(self, env: MultiAgentBattleEnv, step: int, actions: dict = None) -> dict:
        """FrameDTO 형식으로 데이터 구성"""

        # 팀별 유닛 정보
        team_a = []
        team_b = []

        for agent_id in env.team_a:
            unit = env.units[agent_id]
            team_a.append({
                "x": unit.x,
                "y": unit.y,
                "hp": unit.hp,
                "maxHp": unit.max_hp,
                "mp": unit.mp,
                "maxMp": unit.max_mp,
                "role": int(unit.role),
                "alive": unit.is_alive
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
                "alive": unit.is_alive
            })

        # 타일 맵 (2D -> 1D 직렬화)
        tiles = env.game_map.tiles.flatten().tolist()

        # 승자 정보
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
            "playerIdx": -1  # 관전 모드는 플레이어 없음
        }

        return frame

    def close(self):
        self.sock.close()


def play_with_streaming(env: MultiAgentBattleEnv, streamer: UnityStreamer,
                        agent=None, num_episodes: int = 5,
                        delay: float = 0.1, deterministic: bool = False):
    """Unity로 스트리밍하면서 게임 플레이 (단일 모델)"""

    for ep in range(num_episodes):
        print(f"\n=== Episode {ep + 1}/{num_episodes} ===")
        obs = env.reset()
        done = False
        step = 0

        # 초기 상태 전송
        streamer.send_frame(env, step)
        time.sleep(delay)

        while not done:
            # 행동 선택
            if agent is not None:
                actions, _, _ = agent.get_actions(obs, deterministic=deterministic)
            else:
                # 랜덤 행동
                actions = {aid: np.random.randint(0, env.action_space_size)
                          for aid in env.get_agent_ids()}

            # 환경 스텝
            obs, rewards, dones, truncated, infos = env.step(actions)
            done = all(dones.values())
            step += 1

            # Unity로 전송
            streamer.send_frame(env, step, actions)
            time.sleep(delay)

            if done:
                winner_str = "Team A" if env.winner == 0 else "Team B" if env.winner == 1 else "Draw"
                print(f"Game Over! Winner: {winner_str}, Steps: {step}")
                time.sleep(2)  # 결과 확인 시간


def play_with_dual_models(env: MultiAgentBattleEnv, streamer: UnityStreamer,
                          agent_a, agent_b, num_episodes: int = 5,
                          delay: float = 0.1, deterministic: bool = False,
                          random_player_idx: bool = False,
                          obs_size_a: int = None, obs_size_b: int = None):
    """Unity로 스트리밍하면서 게임 플레이 (팀별 다른 모델)

    v11 협동 모델 호환:
    - obs_size_a가 env obs보다 크면 tank_info(6) 자동 추가
    - obs_size_b가 env obs와 다르면 조정
    """
    env_obs_size = None  # 첫 에피소드에서 설정

    for ep in range(num_episodes):
        # 매 에피소드마다 가상 플레이어 역할 랜덤 선택 (협동 모델 테스트용)
        if random_player_idx:
            env.config.player_idx = np.random.randint(0, 5)
            print(f"\n=== Episode {ep + 1}/{num_episodes} (Virtual Player: {env.config.player_idx}) ===")
        else:
            print(f"\n=== Episode {ep + 1}/{num_episodes} ===")

        obs = env.reset()
        done = False
        step = 0

        # env obs 크기 확인 (첫 에피소드에서만)
        if env_obs_size is None:
            env_obs_size = len(list(obs.values())[0])
            print(f"[Config] env obs_size: {env_obs_size}")
            if obs_size_a and obs_size_a > env_obs_size:
                print(f"[Config] A팀 모델({obs_size_a})이 tank_info({obs_size_a - env_obs_size}) 필요 → 자동 추가")

        # 탱크 ID (협동 모드용)
        tank_id = f"team_a_{env.config.player_idx}" if env.config.player_idx >= 0 else None

        # 초기 상태 전송
        streamer.send_frame(env, step)
        time.sleep(delay)

        while not done:
            # === Team A 행동 선택 ===
            obs_a = {}
            for aid in env.team_a:
                a_ob = obs[aid]

                # obs_size_a가 env보다 크면 tank_info 추가
                if obs_size_a and obs_size_a > env_obs_size and tank_id:
                    tank_info = get_tank_info(env, aid, tank_id)
                    a_ob = np.concatenate([a_ob, tank_info])

                obs_a[aid] = a_ob

            if agent_a is not None:
                actions_a, _, _ = agent_a.get_actions(obs_a, deterministic=deterministic)
            else:
                actions_a = {aid: np.random.randint(0, env.action_space_size)
                            for aid in env.team_a}

            # === Team B 행동 선택 ===
            obs_b = {}
            for aid in env.team_b:
                b_ob = obs[aid]

                # obs_size_b 조정
                if obs_size_b and obs_size_b < env_obs_size:
                    b_ob = b_ob[:obs_size_b]
                elif obs_size_b and obs_size_b > env_obs_size:
                    padding = np.zeros(obs_size_b - env_obs_size, dtype=np.float32)
                    b_ob = np.concatenate([b_ob, padding])

                obs_b[aid] = b_ob

            if agent_b is not None:
                actions_b, _, _ = agent_b.get_actions(obs_b, deterministic=deterministic)
            else:
                actions_b = {aid: np.random.randint(0, env.action_space_size)
                            for aid in env.team_b}

            # 행동 합치기
            actions = {**actions_a, **actions_b}

            # 환경 스텝
            obs, rewards, dones, truncated, infos = env.step(actions)
            done = all(dones.values())
            step += 1

            # Unity로 전송
            streamer.send_frame(env, step, actions)
            time.sleep(delay)

            if done:
                winner_str = "Team A" if env.winner == 0 else "Team B" if env.winner == 1 else "Draw"
                print(f"Game Over! Winner: {winner_str}, Steps: {step}")
                time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description='Stream RL_Game_NPC to Unity')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Unity host')
    parser.add_argument('--port', type=int, default=5005, help='UDP port')
    parser.add_argument('--mode', type=str, default='trained', choices=['random', 'trained', 'asymmetric'],
                        help='Play mode (random, trained, asymmetric)')
    parser.add_argument('--model', type=str, default='models_v3_20m/model_latest.pt',
                        help='Model path (for trained mode, or Team A model for asymmetric)')
    parser.add_argument('--model-b', type=str, default=None,
                        help='Team B model path (for asymmetric mode)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between frames (seconds)')
    parser.add_argument('--deterministic', action='store_true', default=False, help='Use deterministic actions')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    # 스탯 배율 인자
    parser.add_argument('--stat-a', type=float, default=1.0, help='Team A stat multiplier (0.7 = 70%)')
    parser.add_argument('--stat-b', type=float, default=1.0, help='Team B stat multiplier (1.0 = 100%)')
    # 협동 모드 설정 (관전용)
    parser.add_argument('--player-idx', type=int, default=-1,
                        help='Virtual player index for coop model (-1=none, 0=Tank, 1=Dealer, etc.)')
    parser.add_argument('--random-player-idx', action='store_true',
                        help='Randomize virtual player index each episode')

    args = parser.parse_args()

    # 환경 생성 (스탯 배율 적용)
    env_config = EnvConfig()
    env_config.team_a_stat_multiplier = args.stat_a
    env_config.team_b_stat_multiplier = args.stat_b
    env_config.player_idx = args.player_idx  # 협동 모델용 가상 플레이어
    env = MultiAgentBattleEnv(env_config)

    print(f"[Config] Team A stat: {args.stat_a*100:.0f}%, Team B stat: {args.stat_b*100:.0f}%")

    # 스트리머 생성
    streamer = UnityStreamer(args.host, args.port)

    try:
        if args.mode == 'random':
            print("Playing with random agents...")
            play_with_streaming(env, streamer, agent=None,
                              num_episodes=args.episodes, delay=args.delay)

        elif args.mode == 'trained':
            print("Playing with trained agent (same model for both teams)...")
            obs = env.reset(seed=args.seed)
            obs_size = list(obs.values())[0].shape[0]
            action_size = env.action_space_size

            agent = PPOAgent(obs_size, action_size)

            if os.path.exists(args.model):
                agent.load(args.model)
                print(f"Loaded model: {args.model}")
                play_with_streaming(env, streamer, agent=agent,
                                  num_episodes=args.episodes, delay=args.delay,
                                  deterministic=args.deterministic)
            else:
                print(f"Model not found: {args.model}, using random actions")
                play_with_streaming(env, streamer, agent=None,
                                  num_episodes=args.episodes, delay=args.delay)

        elif args.mode == 'asymmetric':
            print("Playing with asymmetric models (different model for each team)...")
            obs = env.reset(seed=args.seed)
            obs_size = list(obs.values())[0].shape[0]
            action_size = env.action_space_size

            # Team A 모델
            agent_a = None
            if args.model and os.path.exists(args.model):
                agent_a = PPOAgent(obs_size, action_size)
                agent_a.load(args.model)
                print(f"Team A model loaded: {args.model}")
            else:
                print(f"Team A model not found: {args.model}, using random")

            # Team B 모델
            agent_b = None
            model_b_path = args.model_b or args.model  # model_b가 없으면 model 사용
            if model_b_path and os.path.exists(model_b_path):
                agent_b = PPOAgent(obs_size, action_size)
                agent_b.load(model_b_path)
                print(f"Team B model loaded: {model_b_path}")
            else:
                print(f"Team B model not found: {model_b_path}, using random")

            play_with_dual_models(env, streamer, agent_a, agent_b,
                                 num_episodes=args.episodes, delay=args.delay,
                                 deterministic=args.deterministic,
                                 random_player_idx=args.random_player_idx)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        streamer.close()
        print("Streamer closed")


if __name__ == '__main__':
    main()
