"""학습 스크립트"""
import os
import sys

# L 드라이브 라이브러리 경로 추가 (CUDA PyTorch)
sys.path.insert(0, "L:/RL_game/libs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
from collections import deque
import numpy as np

from src.config import EnvConfig, TrainConfig
from src.env import MultiAgentBattleEnv
from src.agent import PPOAgent


# 전역 로그 파일 경로
LOG_FILE = None


def log(msg):
    """화면과 파일에 동시에 로그 출력"""
    print(msg)
    if LOG_FILE:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


def train(args):
    """학습 메인 함수"""
    global LOG_FILE
    LOG_FILE = args.log_file

    # 로그 파일 초기화
    if LOG_FILE:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    log("=" * 60)
    log("5vs5 Multi-Agent Battle - Self-Play Training")
    log("=" * 60)

    # 설정
    env_config = EnvConfig()
    train_config = TrainConfig(
        learning_rate=args.lr,
        total_timesteps=args.total_steps,
        num_steps=args.rollout_steps,
        batch_size=args.batch_size,
    )

    # 환경 생성
    env = MultiAgentBattleEnv(env_config)

    # 관찰 크기 계산
    obs = env.reset()
    obs_size = list(obs.values())[0].shape[0]
    action_size = env.action_space_size

    log(f"Observation size: {obs_size}")
    log(f"Action size: {action_size}")
    log(f"Number of agents: {env.num_agents}")

    # 에이전트 생성
    agent = PPOAgent(obs_size, action_size, train_config)

    # 기존 모델 로드
    if args.load_model and os.path.exists(args.load_model):
        agent.load(args.load_model)
        log(f"Loaded model from {args.load_model}")

    # 학습 통계
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    win_counts = {'team_a': 0, 'team_b': 0, 'draw': 0}

    total_steps = 0
    episode = 0
    best_win_rate = 0

    log("\nStarting training...")
    start_time = time.time()

    while total_steps < train_config.total_timesteps:
        # 에피소드 시작
        obs = env.reset()
        episode_reward = {aid: 0 for aid in env.get_agent_ids()}
        episode_length = 0

        while True:
            # 행동 선택
            actions, log_probs, values = agent.get_actions(obs)

            # 환경 스텝
            next_obs, rewards, dones, truncated, infos = env.step(actions)

            # 경험 저장
            for agent_id in env.get_agent_ids():
                agent.store_transition(
                    agent_id,
                    obs[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    values[agent_id],
                    log_probs[agent_id],
                    dones[agent_id]
                )
                episode_reward[agent_id] += rewards[agent_id]

            obs = next_obs
            episode_length += 1
            total_steps += env.num_agents

            # 에피소드 종료
            if all(dones.values()):
                break

            # 롤아웃 완료 시 학습
            if episode_length % train_config.num_steps == 0:
                learn_stats = agent.learn()

        # 에피소드 통계
        avg_reward = np.mean(list(episode_reward.values()))
        episode_rewards.append(avg_reward)
        episode_lengths.append(episode_length)

        # 승리 기록
        winner = infos.get('winner')
        if winner == 0:
            win_counts['team_a'] += 1
        elif winner == 1:
            win_counts['team_b'] += 1
        else:
            win_counts['draw'] += 1

        episode += 1

        # 롤아웃 학습
        learn_stats = agent.learn()

        # 로깅
        if episode % args.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed

            avg_ep_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_ep_length = np.mean(episode_lengths) if episode_lengths else 0

            draw_rate = 100 * win_counts['draw'] / episode

            log(f"\n[Episode {episode}] Steps: {total_steps:,}")
            log(f"  Avg Reward: {avg_ep_reward:.2f}")
            log(f"  Avg Length: {avg_ep_length:.1f}")
            log(f"  Win Rate A: {win_counts['team_a']}/{episode} ({100*win_counts['team_a']/episode:.1f}%)")
            log(f"  Win Rate B: {win_counts['team_b']}/{episode} ({100*win_counts['team_b']/episode:.1f}%)")
            log(f"  Draws: {win_counts['draw']} ({draw_rate:.1f}%)")
            log(f"  FPS: {fps:.0f}")

            if learn_stats:
                log(f"  Policy Loss: {learn_stats.get('policy_loss', 0):.4f}")
                log(f"  Value Loss: {learn_stats.get('value_loss', 0):.4f}")
                log(f"  Entropy: {learn_stats.get('entropy', 0):.4f}")

        # 모델 저장
        if episode % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"model_ep{episode}.pt")
            agent.save(save_path)
            log(f"Model saved to {save_path}")

            # 최신 모델도 저장
            latest_path = os.path.join(args.save_dir, "model_latest.pt")
            agent.save(latest_path)

    # 최종 모델 저장
    final_path = os.path.join(args.save_dir, "model_final.pt")
    agent.save(final_path)

    log("\n" + "=" * 60)
    log("Training Complete!")
    log(f"Total Episodes: {episode}")
    log(f"Total Steps: {total_steps:,}")
    log(f"Time: {time.time() - start_time:.1f}s")
    log(f"Final Win Rate A: {100*win_counts['team_a']/episode:.1f}%")
    log(f"Final Win Rate B: {100*win_counts['team_b']/episode:.1f}%")
    log(f"Final Draw Rate: {100*win_counts['draw']/episode:.1f}%")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train 5vs5 Multi-Agent Battle')

    # 학습 설정
    parser.add_argument('--total-steps', type=int, default=500000,
                        help='Total training steps')
    parser.add_argument('--rollout-steps', type=int, default=128,
                        help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')

    # 저장/로드
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Model save directory')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save interval (episodes)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log interval (episodes)')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Load model path')
    parser.add_argument('--log-file', type=str, default='training_log.txt',
                        help='Log file path')

    args = parser.parse_args()

    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)

    train(args)


if __name__ == '__main__':
    main()