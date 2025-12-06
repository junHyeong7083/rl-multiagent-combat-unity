"""
협동 학습 진행도 테스트 스크립트
- A팀: 완전 랜덤 정책 vs 학습된 협동 모델
- B팀: 학습된 모델 (동일)
- 학습 중인 프로세스에 영향 없음
"""
import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env import MultiAgentBattleEnv
from src.config import EnvConfig
from src.agent import PPOAgent


def run_random_vs_trained(env, agent_trained, num_episodes=100):
    """A팀 랜덤 vs B팀 학습된 A모델 (A모델 강도 테스트)"""
    results = {'a_win': 0, 'b_win': 0, 'draw': 0}

    for ep in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            # A팀: 완전 랜덤
            actions_a = {aid: np.random.randint(0, env.action_space_size) for aid in env.team_a}

            # B팀: 학습된 A모델 사용 (A모델이 얼마나 강해졌는지 확인)
            obs_b = {aid: obs[aid] for aid in env.team_b}
            actions_b, _, _ = agent_trained.get_actions(obs_b, deterministic=True)

            actions = {**actions_a, **actions_b}
            obs, rewards, dones, _, info = env.step(actions)
            done = all(dones.values())

        if info.get('winner') == 0:
            results['a_win'] += 1
        elif info.get('winner') == 1:
            results['b_win'] += 1
        else:
            results['draw'] += 1

    return results


def run_trained_vs_trained(env, agent_a, agent_b, num_episodes=100, player_idx=-1):
    """A팀 학습 모델 vs B팀 학습 모델 (협동 모드)"""
    results = {'a_win': 0, 'b_win': 0, 'draw': 0}
    coop_stats = defaultdict(list)  # 협동 통계

    for ep in range(num_episodes):
        # 플레이어 역할 랜덤 설정
        if player_idx < 0:
            env.config.player_idx = np.random.randint(0, 5)
        else:
            env.config.player_idx = player_idx

        obs = env.reset()
        done = False

        # 협동 거리 추적
        player_agent_id = f"team_a_{env.config.player_idx}"
        distances_to_player = []

        while not done:
            # A팀: 학습 모델
            obs_a = {aid: obs[aid] for aid in env.team_a}
            actions_a, _, _ = agent_a.get_actions(obs_a, deterministic=True)

            # B팀: 학습 모델
            obs_b = {aid: obs[aid] for aid in env.team_b}
            actions_b, _, _ = agent_b.get_actions(obs_b, deterministic=True)

            actions = {**actions_a, **actions_b}
            obs, rewards, dones, _, info = env.step(actions)
            done = all(dones.values())

            # 플레이어와 AI 팀원 간 거리 측정
            if player_agent_id in env.units:
                player_unit = env.units[player_agent_id]
                if player_unit.is_alive:
                    for aid in env.team_a:
                        if aid != player_agent_id and aid in env.units:
                            ai_unit = env.units[aid]
                            if ai_unit.is_alive:
                                dist = player_unit.distance_to(ai_unit)
                                distances_to_player.append(dist)

        if info.get('winner') == 0:
            results['a_win'] += 1
        elif info.get('winner') == 1:
            results['b_win'] += 1
        else:
            results['draw'] += 1

        if distances_to_player:
            coop_stats['avg_dist'].append(np.mean(distances_to_player))

    return results, coop_stats


def main():
    parser = argparse.ArgumentParser(description='협동 학습 진행도 테스트')
    parser.add_argument('--model-a', type=str, default='models_coop_v1/model_a_latest.pt',
                        help='A팀 모델 경로')
    parser.add_argument('--model-b', type=str, default='models_coop_v1/model_b_latest.pt',
                        help='B팀 모델 경로')
    parser.add_argument('--episodes', type=int, default=100,
                        help='테스트 에피소드 수')
    parser.add_argument('--player-idx', type=int, default=-1,
                        help='플레이어 인덱스 (-1: 랜덤)')
    args = parser.parse_args()

    print("=" * 60)
    print("협동 학습 진행도 테스트")
    print("=" * 60)

    # 환경 생성 (학습과 독립)
    config = EnvConfig()
    config.player_idx = args.player_idx if args.player_idx >= 0 else 0
    env = MultiAgentBattleEnv(config)

    # 관측 크기 확인
    obs = env.reset()
    obs_size = list(obs.values())[0].shape[0]
    action_size = env.action_space_size

    print(f"\n[환경] obs_size={obs_size}, action_size={action_size}")
    print(f"[모델] A팀: {args.model_a}")
    print(f"[모델] B팀: {args.model_b}")

    # 에이전트 로드
    agent_a = PPOAgent(obs_size, action_size)
    agent_b = PPOAgent(obs_size, action_size)

    model_a_exists = os.path.exists(args.model_a)
    model_b_exists = os.path.exists(args.model_b)

    if model_a_exists:
        agent_a.load(args.model_a)
        print(f"[로드] A팀 모델 로드 완료")
    else:
        print(f"[경고] A팀 모델 없음 - 랜덤 정책 사용")

    if model_b_exists:
        agent_b.load(args.model_b)
        print(f"[로드] B팀 모델 로드 완료")
    else:
        print(f"[경고] B팀 모델 없음 - 랜덤 정책 사용")

    # === 테스트 1: 랜덤 A vs 학습된 A모델(B팀에 배치) ===
    print(f"\n{'='*60}")
    print("테스트 1: 랜덤 A팀 vs 학습된 A모델 (B팀에 배치)")
    print("(A팀 모델이 얼마나 강해졌는지 확인)")
    print("=" * 60)

    env.config.player_idx = -1  # 협동 모드 끄기
    results1 = run_random_vs_trained(env, agent_a, args.episodes)

    print(f"\n[결과] {args.episodes}판 기준")
    print(f"  랜덤 A팀 승리: {results1['a_win']} ({100*results1['a_win']/args.episodes:.1f}%)")
    print(f"  학습 A모델(B팀) 승리: {results1['b_win']} ({100*results1['b_win']/args.episodes:.1f}%)")
    print(f"  무승부: {results1['draw']} ({100*results1['draw']/args.episodes:.1f}%)")

    if results1['b_win'] > results1['a_win']:
        print(f"\n  ★ A모델이 랜덤보다 {results1['b_win'] - results1['a_win']}판 더 이김! 학습 효과 있음!")
    else:
        print(f"\n  → 아직 학습 효과 미미 (더 학습 필요)")

    # === 테스트 2: 학습된 A vs 학습된 B (협동 모드) ===
    if model_a_exists:
        print(f"\n{'='*60}")
        print("테스트 2: 학습된 A팀 vs 학습된 B팀 (협동 모드)")
        print("(A팀이 플레이어와 얼마나 협동하는지 확인)")
        print("=" * 60)

        results2, coop_stats = run_trained_vs_trained(
            env, agent_a, agent_b, args.episodes, args.player_idx
        )

        print(f"\n[결과] {args.episodes}판 기준")
        print(f"  협동 A팀 승리: {results2['a_win']} ({100*results2['a_win']/args.episodes:.1f}%)")
        print(f"  적 B팀 승리: {results2['b_win']} ({100*results2['b_win']/args.episodes:.1f}%)")
        print(f"  무승부: {results2['draw']} ({100*results2['draw']/args.episodes:.1f}%)")

        if coop_stats['avg_dist']:
            avg_dist = np.mean(coop_stats['avg_dist'])
            print(f"\n[협동 지표]")
            print(f"  AI-플레이어 평균 거리: {avg_dist:.2f}")
            if avg_dist < 3:
                print(f"  → 협동 우수! (거리 3 이하)")
            elif avg_dist < 5:
                print(f"  → 협동 보통 (거리 5 이하)")
            else:
                print(f"  → 협동 필요 (거리 5 초과)")

    # === 테스트 3: 랜덤 A vs 랜덤 B (기준선) ===
    print(f"\n{'='*60}")
    print("테스트 3: 랜덤 A팀 vs 랜덤 B팀 (기준선)")
    print("=" * 60)

    env.config.player_idx = -1
    results3 = {'a_win': 0, 'b_win': 0, 'draw': 0}

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            actions = {aid: np.random.randint(0, env.action_space_size)
                      for aid in env.get_agent_ids()}
            obs, _, dones, _, info = env.step(actions)
            done = all(dones.values())

        if info.get('winner') == 0:
            results3['a_win'] += 1
        elif info.get('winner') == 1:
            results3['b_win'] += 1
        else:
            results3['draw'] += 1

    print(f"\n[결과] {args.episodes}판 기준")
    print(f"  랜덤 A팀 승리: {results3['a_win']} ({100*results3['a_win']/args.episodes:.1f}%)")
    print(f"  랜덤 B팀 승리: {results3['b_win']} ({100*results3['b_win']/args.episodes:.1f}%)")
    print(f"  무승부: {results3['draw']} ({100*results3['draw']/args.episodes:.1f}%)")

    # === 종합 분석 ===
    print(f"\n{'='*60}")
    print("종합 분석")
    print("=" * 60)

    b_improvement = results1['b_win'] - results3['b_win']
    print(f"\n[B팀 학습 효과]")
    print(f"  랜덤 대비 승률 변화: {b_improvement:+d}판")

    if model_a_exists:
        a_vs_random = results2['a_win'] - results3['a_win']
        print(f"\n[A팀 협동 효과]")
        print(f"  랜덤 대비 승률 변화: {a_vs_random:+d}판")

        if results2['a_win'] > results1['a_win']:
            print(f"\n  ★ 협동 학습이 효과적! (랜덤보다 {results2['a_win']-results1['a_win']}판 더 이김)")
        elif results2['a_win'] == results1['a_win']:
            print(f"\n  → 아직 협동 효과 없음 (더 학습 필요)")
        else:
            print(f"\n  → 학습 초기 단계 (더 학습 필요)")


if __name__ == '__main__':
    main()
