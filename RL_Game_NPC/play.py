"""테스트/시각화 스크립트"""
import os
import sys

# L 드라이브 라이브러리 경로 추가 (CUDA PyTorch)
sys.path.insert(0, "L:/RL_game/libs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import numpy as np

from src.config import EnvConfig, VisualConfig
from src.env import MultiAgentBattleEnv
from src.agent import PPOAgent
from src.visualizer import GameVisualizer


def play_random(env, visualizer, num_episodes=5):
    """랜덤 에이전트로 플레이"""
    print("Playing with random agents...")

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1} ---")
        obs = env.reset()
        done = False
        step = 0

        while not done:
            # 랜덤 행동
            actions = {aid: np.random.randint(0, env.action_space_size)
                      for aid in env.get_agent_ids()}

            obs, rewards, dones, truncated, infos = env.step(actions)
            done = all(dones.values())
            step += 1

            # 렌더링
            state = env.render()
            if not visualizer.render(state):
                return

            # 게임 종료 시 대기
            if done:
                print(f"Game Over! Winner: {'Team A' if infos['winner'] == 0 else 'Team B' if infos['winner'] == 1 else 'Draw'}")
                print(f"Steps: {step}")
                time.sleep(2)


def play_trained(env, visualizer, agent, num_episodes=5, deterministic=False):
    """학습된 에이전트로 플레이"""
    print("Playing with trained agent...")

    wins = {'team_a': 0, 'team_b': 0, 'draw': 0}

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1} ---")
        obs = env.reset()
        done = False
        step = 0
        total_rewards = {aid: 0 for aid in env.get_agent_ids()}

        while not done:
            # 학습된 정책으로 행동 선택
            actions, _, _ = agent.get_actions(obs, deterministic=deterministic)

            obs, rewards, dones, truncated, infos = env.step(actions)
            done = all(dones.values())
            step += 1

            for aid, r in rewards.items():
                total_rewards[aid] += r

            # 렌더링
            state = env.render()
            if not visualizer.render(state):
                return wins

            # 게임 종료 시 대기
            if done:
                winner = infos.get('winner')
                winner_str = 'Team A' if winner == 0 else 'Team B' if winner == 1 else 'Draw'
                print(f"Game Over! Winner: {winner_str}")
                print(f"Steps: {step}")
                print(f"Avg Reward: {np.mean(list(total_rewards.values())):.2f}")

                if winner == 0:
                    wins['team_a'] += 1
                elif winner == 1:
                    wins['team_b'] += 1
                else:
                    wins['draw'] += 1

                time.sleep(2)

    return wins


def play_interactive(env, visualizer):
    """수동 조작 모드 (Team A만)"""
    print("\nInteractive Mode (Control Team A)")
    print("Controls:")
    print("  Arrow Keys: Move selected unit")
    print("  1-5: Select unit")
    print("  Space: Attack nearest enemy")
    print("  Q: AOE skill")
    print("  W: Heal skill")
    print("  Enter: End turn")
    print("  ESC: Quit")

    import pygame

    obs = env.reset()
    done = False
    selected_unit = 0  # 0-4

    while not done:
        # Team A 행동 입력 대기
        team_a_actions = {f"team_a_{i}": 0 for i in range(5)}  # 기본: 제자리

        waiting_input = True
        while waiting_input:
            state = env.render()
            if not visualizer.render(state):
                return

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return

                    # 유닛 선택
                    if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        selected_unit = event.key - pygame.K_1
                        print(f"Selected unit {selected_unit}")

                    # 이동
                    aid = f"team_a_{selected_unit}"
                    if event.key == pygame.K_UP:
                        team_a_actions[aid] = 1
                    elif event.key == pygame.K_DOWN:
                        team_a_actions[aid] = 2
                    elif event.key == pygame.K_LEFT:
                        team_a_actions[aid] = 3
                    elif event.key == pygame.K_RIGHT:
                        team_a_actions[aid] = 4

                    # 공격/스킬
                    elif event.key == pygame.K_SPACE:
                        team_a_actions[aid] = 5  # 가까운 적 공격
                    elif event.key == pygame.K_q:
                        team_a_actions[aid] = 7  # AOE
                    elif event.key == pygame.K_w:
                        team_a_actions[aid] = 8  # 힐

                    # 턴 종료
                    elif event.key == pygame.K_RETURN:
                        waiting_input = False

        # Team B는 랜덤
        team_b_actions = {f"team_b_{i}": np.random.randint(0, env.action_space_size)
                        for i in range(5)}

        actions = {**team_a_actions, **team_b_actions}
        obs, rewards, dones, truncated, infos = env.step(actions)
        done = all(dones.values())

        if done:
            winner = infos.get('winner')
            winner_str = 'Team A (You)' if winner == 0 else 'Team B' if winner == 1 else 'Draw'
            print(f"\nGame Over! Winner: {winner_str}")
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description='Play 5vs5 Multi-Agent Battle')

    parser.add_argument('--mode', type=str, default='random',
                        choices=['random', 'trained', 'interactive'],
                        help='Play mode')
    parser.add_argument('--model', type=str, default='models/model_latest.pt',
                        help='Model path for trained mode')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic actions')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    args = parser.parse_args()

    # 설정
    env_config = EnvConfig()
    visual_config = VisualConfig(fps=args.fps)

    # 환경 생성
    env = MultiAgentBattleEnv(env_config)

    # 시각화 생성
    visualizer = GameVisualizer(env_config, visual_config)

    try:
        if args.mode == 'random':
            play_random(env, visualizer, args.episodes)

        elif args.mode == 'trained':
            # 에이전트 생성 및 로드
            obs = env.reset(seed=args.seed)
            obs_size = list(obs.values())[0].shape[0]
            action_size = env.action_space_size

            agent = PPOAgent(obs_size, action_size)

            if os.path.exists(args.model):
                agent.load(args.model)
                wins = play_trained(env, visualizer, agent, args.episodes, args.deterministic)
                print(f"\nResults: Team A: {wins['team_a']}, Team B: {wins['team_b']}, Draw: {wins['draw']}")
            else:
                print(f"Model not found: {args.model}")
                print("Running with random actions instead...")
                play_random(env, visualizer, args.episodes)

        elif args.mode == 'interactive':
            play_interactive(env, visualizer)

    finally:
        visualizer.close()


if __name__ == '__main__':
    main()