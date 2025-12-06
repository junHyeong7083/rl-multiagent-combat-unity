"""환경 테스트 스크립트"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from src.config import EnvConfig
from src.env import MultiAgentBattleEnv

def test_environment():
    print("Testing MultiAgentBattleEnv...")
    print("=" * 50)

    # 환경 생성
    config = EnvConfig()
    env = MultiAgentBattleEnv(config)

    # 리셋 테스트
    print("\n1. Testing reset()...")
    obs = env.reset(seed=42)
    print(f"   Number of agents: {len(obs)}")
    print(f"   Observation shape: {list(obs.values())[0].shape}")
    print(f"   Agent IDs: {list(obs.keys())}")

    # 스텝 테스트
    print("\n2. Testing step()...")
    for step in range(10):
        # 랜덤 행동
        actions = {aid: np.random.randint(0, env.action_space_size)
                  for aid in env.get_agent_ids()}

        obs, rewards, dones, truncated, infos = env.step(actions)

        if step == 0:
            print(f"   Rewards sample: {list(rewards.values())[:3]}")
            print(f"   Infos: {infos}")

        if all(dones.values()):
            print(f"   Game ended at step {step + 1}")
            print(f"   Winner: {'Team A' if infos['winner'] == 0 else 'Team B' if infos['winner'] == 1 else 'Draw'}")
            break

    # 유닛 정보 확인
    print("\n3. Unit Information:")
    for agent_id, unit in env.units.items():
        print(f"   {agent_id}: {unit.role.name}, HP={unit.hp}/{unit.max_hp}, pos=({unit.x},{unit.y})")

    # 맵 정보 확인
    print("\n4. Map Information:")
    print(f"   Map size: {env.game_map.width}x{env.game_map.height}")
    unique, counts = np.unique(env.game_map.tiles, return_counts=True)
    from src.config import TileType
    for tile_type, count in zip(unique, counts):
        print(f"   {TileType(tile_type).name}: {count} tiles")

    # 전체 에피소드 테스트
    print("\n5. Running full episode...")
    obs = env.reset()
    total_rewards = {aid: 0 for aid in env.get_agent_ids()}
    steps = 0

    while True:
        actions = {aid: np.random.randint(0, env.action_space_size)
                  for aid in env.get_agent_ids()}
        obs, rewards, dones, truncated, infos = env.step(actions)

        for aid, r in rewards.items():
            total_rewards[aid] += r

        steps += 1

        if all(dones.values()):
            break

    print(f"   Episode finished in {steps} steps")
    print(f"   Winner: {'Team A' if infos['winner'] == 0 else 'Team B' if infos['winner'] == 1 else 'Draw'}")
    print(f"   Team A alive: {infos['team_a_alive']}")
    print(f"   Team B alive: {infos['team_b_alive']}")
    print(f"   Avg total reward: {np.mean(list(total_rewards.values())):.2f}")

    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    test_environment()