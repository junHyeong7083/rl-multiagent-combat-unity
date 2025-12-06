"""에이전트 테스트 스크립트"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from src.config import EnvConfig, TrainConfig
from src.env import MultiAgentBattleEnv
from src.agent import PPOAgent


def test_agent():
    print("Testing PPO Agent...")
    print("=" * 50)

    # 환경 생성
    env_config = EnvConfig()
    env = MultiAgentBattleEnv(env_config)

    # 관찰 크기 확인
    obs = env.reset()
    obs_size = list(obs.values())[0].shape[0]
    action_size = env.action_space_size

    print(f"Observation size: {obs_size}")
    print(f"Action size: {action_size}")

    # 에이전트 생성
    train_config = TrainConfig(batch_size=64)
    agent = PPOAgent(obs_size, action_size, train_config)
    print(f"Device: {agent.device}")

    # 행동 선택 테스트
    print("\n1. Testing action selection...")
    actions, log_probs, values = agent.get_actions(obs)
    print(f"   Actions: {list(actions.values())[:5]}")
    print(f"   Log probs: {list(log_probs.values())[:3]}")
    print(f"   Values: {list(values.values())[:3]}")

    # 짧은 학습 테스트
    print("\n2. Testing training loop (3 episodes)...")

    for ep in range(3):
        obs = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            actions, log_probs, values = agent.get_actions(obs)
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
                episode_reward += rewards[agent_id]

            obs = next_obs
            steps += 1

            if all(dones.values()):
                break

        print(f"   Episode {ep + 1}: {steps} steps, reward={episode_reward:.2f}, winner={'A' if infos['winner']==0 else 'B' if infos['winner']==1 else 'Draw'}")

    # 학습 실행
    print("\n3. Testing learning...")
    learn_stats = agent.learn()
    print(f"   Policy loss: {learn_stats.get('policy_loss', 0):.4f}")
    print(f"   Value loss: {learn_stats.get('value_loss', 0):.4f}")
    print(f"   Entropy: {learn_stats.get('entropy', 0):.4f}")

    # 모델 저장/로드 테스트
    print("\n4. Testing save/load...")
    os.makedirs("test_models", exist_ok=True)
    agent.save("test_models/test_model.pt")

    agent2 = PPOAgent(obs_size, action_size, train_config)
    agent2.load("test_models/test_model.pt")
    print("   Save/Load successful!")

    # 정리
    import shutil
    shutil.rmtree("test_models", ignore_errors=True)

    print("\n" + "=" * 50)
    print("All agent tests passed!")


if __name__ == '__main__':
    test_agent()
