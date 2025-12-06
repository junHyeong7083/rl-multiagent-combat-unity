"""협동 강화 학습 스크립트 v11 - 거리 비례 공격 보상

v10 대비 핵심 변경:
1. 거리 비례 공격 보상: 탱크 가까울수록 높은 보상!
   - 0칸: +15, 2칸: +10, 4칸: +5
   - 5칸+: -5 강한 패널티 (탱크 무시하고 공격 방지!)
2. 따라가기 스케일 완화: 0.5 → 0.3 (싸울 여유 확보)
3. Model 정책 비율 감소: 35% → 15% (학습 혼란 방지)
4. Goal 정책 비율 증가: 35% → 50% (따라가기 기본기 강화)

목표: "탱크 근처에서 싸우기"
- v10 문제: 따라가기만 하고 공격 안함
- v11 해결: 탱크 근처 공격 시 높은 보상, 멀리서 공격 시 패널티
"""
import os
import sys
import random

sys.path.insert(0, "L:/RL_game/libs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
from collections import deque
import numpy as np
import torch

from src.config import EnvConfig, TrainConfig, RoleType, ActionType
from src.env import MultiAgentBattleEnv
from src.agent import PPOAgent


LOG_FILE = None
CSV_FILE = None

# 탱커 고정 (인덱스 0)
PLAYER_IDX = 0  # 탱커


def log(msg):
    print(msg)
    if LOG_FILE:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


def init_csv(csv_path):
    headers = ['episode', 'steps', 'avg_reward', 'avg_length',
               'win_rate_a', 'wins_a', 'wins_b', 'draws', 'fps',
               'avg_tank_dist', 'tank_policy_type']
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(','.join(headers) + '\n')


def csv_log(episode, steps, avg_reward, avg_length, win_counts, fps,
            avg_tank_dist=0, policy_type=''):
    if not CSV_FILE:
        return
    total = win_counts['team_a'] + win_counts['team_b'] + win_counts['draw']
    if total == 0:
        total = 1
    win_rate_a = 100 * win_counts['team_a'] / total
    row = [episode, steps, f'{avg_reward:.4f}', f'{avg_length:.2f}',
           f'{win_rate_a:.2f}', win_counts['team_a'], win_counts['team_b'],
           win_counts['draw'], f'{fps:.2f}', f'{avg_tank_dist:.2f}', policy_type]
    with open(CSV_FILE, 'a', encoding='utf-8') as f:
        f.write(','.join(map(str, row)) + '\n')


def get_tank_info(env, npc_id, tank_id):
    """NPC observation에 추가할 탱커 정보 (6차원)

    구조:
    - [0] 상대 x 위치 (정규화)
    - [1] 상대 y 위치 (정규화)
    - [2] 거리 (정규화)
    - [3] 탱커 생존 여부
    - [4] 탱커 HP 비율
    - [5] 탱커 존재 플래그 (항상 1.0)
    """
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


# =============================================================================
# Tank Policies
# =============================================================================

class GoalTankPolicy:
    """Type A: 목표 기반 이동 - 순수 따라가기 학습용"""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.goal = None
        self.map_width = 20
        self.map_height = 20
        self.type_name = "Goal"

    def reset_episode(self, env=None):
        if env:
            self.map_width = env.config.map_width
            self.map_height = env.config.map_height
        self.goal = (
            self.rng.randint(2, self.map_width - 3),
            self.rng.randint(2, self.map_height - 3)
        )

    def _pick_new_goal(self):
        self.goal = (
            self.rng.randint(2, self.map_width - 3),
            self.rng.randint(2, self.map_height - 3)
        )

    def get_action(self, env, tank_id):
        tank = env.units.get(tank_id)
        if tank is None or not tank.is_alive:
            return ActionType.STAY

        if self.goal is None:
            self._pick_new_goal()

        dx = self.goal[0] - tank.x
        dy = self.goal[1] - tank.y

        if abs(dx) <= 1 and abs(dy) <= 1:
            self._pick_new_goal()
            dx = self.goal[0] - tank.x
            dy = self.goal[1] - tank.y

        if abs(dx) > abs(dy):
            return ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
        else:
            return ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP


class ModelTankPolicy:
    """Type B: v11 모델 사용 - 전투 + 따라가기 (적에게 가면 탱크도 감)"""

    def __init__(self, model_path, obs_size, action_size, device='cuda'):
        self.agent = PPOAgent(obs_size, action_size)
        self.agent.load(model_path)
        self.type_name = "Model"

    def reset_episode(self, env=None):
        pass  # 모델 기반이라 리셋 불필요

    def get_action(self, env, tank_id):
        tank = env.units.get(tank_id)
        if tank is None or not tank.is_alive:
            return ActionType.STAY

        # 탱커의 obs 구성 (env._get_observations 호출)
        obs = env._get_observations()[tank_id]
        obs_dict = {tank_id: obs}

        actions, _, _ = self.agent.get_actions(obs_dict, deterministic=False)
        return actions[tank_id]


class ConditionalTankPolicy:
    """Type C: 조건부 - 적 멀면 랜덤 이동, 가까우면 공격"""

    def __init__(self, seed=None, engage_range=4):
        self.rng = random.Random(seed)
        self.goal = None
        self.map_width = 20
        self.map_height = 20
        self.engage_range = engage_range  # 이 거리 이내면 공격
        self.type_name = "Conditional"

    def reset_episode(self, env=None):
        if env:
            self.map_width = env.config.map_width
            self.map_height = env.config.map_height
        self._pick_new_goal()

    def _pick_new_goal(self):
        self.goal = (
            self.rng.randint(2, self.map_width - 3),
            self.rng.randint(2, self.map_height - 3)
        )

    def get_action(self, env, tank_id):
        tank = env.units.get(tank_id)
        if tank is None or not tank.is_alive:
            return ActionType.STAY

        # 가장 가까운 적 확인
        enemies = [u for u in env.units.values()
                   if u.team_id != tank.team_id and u.is_alive]

        if enemies:
            nearest = min(enemies, key=lambda e: tank.distance_to(e))
            dist_to_enemy = tank.distance_to(nearest)

            # 적이 가까우면 공격
            if dist_to_enemy <= self.engage_range:
                # 공격 범위 내면 공격
                if dist_to_enemy <= 1:  # 탱커 공격 범위
                    return ActionType.ATTACK_NEAREST
                # 적에게 접근
                dx = nearest.x - tank.x
                dy = nearest.y - tank.y
                if abs(dx) > abs(dy):
                    return ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
                else:
                    return ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP

        # 적이 멀면 목표점으로 이동
        if self.goal is None:
            self._pick_new_goal()

        dx = self.goal[0] - tank.x
        dy = self.goal[1] - tank.y

        if abs(dx) <= 1 and abs(dy) <= 1:
            self._pick_new_goal()
            dx = self.goal[0] - tank.x
            dy = self.goal[1] - tank.y

        if abs(dx) > abs(dy):
            return ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
        else:
            return ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP


class MixedTankPolicy:
    """여러 정책을 혼합 - 에피소드마다 랜덤 선택"""

    def __init__(self, policies, weights=None, seed=None):
        """
        Args:
            policies: [(policy_instance, weight), ...] 또는 [policy, ...]
            weights: 각 정책의 선택 확률 (None이면 균등)
        """
        self.policies = policies
        self.weights = weights or [1.0] * len(policies)
        # 정규화
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        self.rng = random.Random(seed)
        self.current_policy = None
        self.type_name = "Mixed"

    def reset_episode(self, env=None):
        """에피소드 시작 시 정책 선택"""
        self.current_policy = self.rng.choices(self.policies, self.weights)[0]
        self.current_policy.reset_episode(env)
        self.type_name = f"Mixed({self.current_policy.type_name})"

    def get_action(self, env, tank_id):
        if self.current_policy is None:
            self.reset_episode(env)
        return self.current_policy.get_action(env, tank_id)


# =============================================================================
# Config & Rewards
# =============================================================================

def apply_v11_config(env_config):
    """v11 설정: 공격 보상 강화 + 따라가기 완화"""
    env_config.player_idx = PLAYER_IDX

    # v11 플래그
    env_config.use_v7_simple = True

    # === 탱크 따라가기 보상 (v10 대비 완화) ===
    env_config.reward_per_tile = 2.5        # v10: 3.0 → v11: 2.5
    env_config.proximity_threshold = 2      # 2칸 이내 = 보상

    # 이탈 패널티 (완화!)
    env_config.reward_leave_per_tile = -1.5  # v10: -2.0 → v11: -1.5

    # === 전투 보상 강화! (핵심 변경) ===
    env_config.reward_combat = 15.0       # v10: 5.0 → v11: 15.0 (3배 증가!)
    env_config.combat_range = 4           # v10: 3칸 → v11: 4칸 (범위 확대)
    env_config.reward_far_attack = -1.0   # v10: -2.0 → v11: -1.0 (패널티 완화)

    # 기본 전투 보상 (강화!)
    env_config.reward_kill = 15.0         # v10: 10.0 → v11: 15.0
    env_config.reward_damage = 0.5        # v10: 0.3 → v11: 0.5
    env_config.reward_win = 25.0          # v10: 20.0 → v11: 25.0
    env_config.reward_lose = -15.0
    env_config.reward_draw = -10.0

    # 탱커 뒤 위치 (완화)
    env_config.reward_behind_tank = 0.2   # v10: 0.3 → v11: 0.2

    # 시간 패널티
    env_config.reward_time_penalty = -0.02  # v10: -0.03 → v11: -0.02

    return env_config


def calculate_follow_reward(env, npc_ids, tank_id, prev_distances,
                            actions=None, rewards_env=None):
    """탱크 따라가기 보상 계산 (v11: 거리 비례 공격 보상)

    v10 대비 변경:
    - scale: 0.5 → 0.3 (따라가기 압박 완화)
    - 공격 보상: 거리 비례! (0칸: +15, 2칸: +10, 4칸: +5)
    - 멀리서 공격: -5 강한 패널티 (탱크 무시 방지!)

    목표: "탱크 근처에서 싸우기" - 따라가면서 공격해야 최대 보상
    """
    rewards = {}
    current_distances = {}

    tank = env.units.get(tank_id)
    if tank is None or not tank.is_alive:
        return {aid: 0.0 for aid in npc_ids}, prev_distances

    # 거리 비례 보상 파라미터 (v11 조정!)
    base_reward = 4.0   # 0칸일 때 최대 보상 (유지)
    scale = 0.3         # v10: 0.5 → v11: 0.3 (완화!)

    for aid in npc_ids:
        npc = env.units.get(aid)
        if npc is None or not npc.is_alive:
            rewards[aid] = 0.0
            current_distances[aid] = prev_distances.get(aid, 10.0)
            continue

        dist = npc.distance_to(tank)
        current_distances[aid] = dist
        prev_dist = prev_distances.get(aid, dist)

        reward = 0.0

        # 1. 거리 비례 연속 보상 (완화된 스케일!)
        # v11: dist=0: +4, dist=4: +2.8, dist=8: +1.6, dist=13: 0, dist=20: -2
        reward += base_reward - scale * dist

        # 2. 접근/이탈 보상 (완화!)
        if dist < prev_dist:
            reward += 0.8 * (prev_dist - dist)  # v10: 1.0 → v11: 0.8
        elif dist > prev_dist:
            reward -= 1.0 * (dist - prev_dist)  # v10: 1.5 → v11: 1.0

        # 3. 공격 보상/패널티 (거리 비례! 탱크 근처에서만 높은 보상)
        if actions and rewards_env:
            action = actions.get(aid, 0)
            env_reward = rewards_env.get(aid, 0)

            # 공격 액션 (5=ATTACK_NEAREST, 6=ATTACK_LOWEST, 7~11=스킬)
            is_attack = action >= 5

            if is_attack and env_reward > 0:  # 데미지를 줬다면
                if dist <= 4:
                    # 거리 비례 공격 보상: 가까울수록 높음!
                    # 0칸: +15, 1칸: +12.5, 2칸: +10, 3칸: +7.5, 4칸: +5
                    reward += 15.0 - 2.5 * dist
                else:
                    # 5칸 이상에서 공격: 강한 패널티! (탱크 무시 방지)
                    reward -= 5.0

        rewards[aid] = reward

    return rewards, current_distances


# =============================================================================
# Episode & Training
# =============================================================================

def run_episode(env, agent_npc, agent_b, tank_policy, args):
    """에피소드 실행: 혼합 탱크 정책 + NPC 따라가기 학습"""
    env.config.player_idx = PLAYER_IDX

    obs = env.reset()
    episode_reward = {aid: 0 for aid in env.get_agent_ids()}
    episode_length = 0

    transitions_npc = []

    tank_id = f"team_a_{PLAYER_IDX}"
    npc_ids = [aid for aid in env.team_a if aid != tank_id]

    # 탱크 정책 초기화 (에피소드마다 정책 선택됨)
    tank_policy.reset_episode(env)

    prev_distances = {}
    tank_distances = []

    while True:
        actions = {}

        # === 탱커: Mixed Policy ===
        tank_action = tank_policy.get_action(env, tank_id)
        actions[tank_id] = int(tank_action)

        # === NPC 4명: 탱커 정보 포함 obs (235차원) ===
        obs_npc_extended = {}
        for aid in npc_ids:
            tank_info = get_tank_info(env, aid, tank_id)
            obs_npc_extended[aid] = np.concatenate([obs[aid], tank_info])

        actions_npc, log_probs_npc, values_npc = agent_npc.get_actions(obs_npc_extended)

        # === Team B (고정 모델) ===
        obs_b = {aid: obs[aid] for aid in env.team_b}
        actions_b, _, _ = agent_b.get_actions(obs_b)

        # 모든 액션 합치기
        all_actions = {**{tank_id: int(tank_action)}, **actions_npc, **actions_b}
        next_obs, rewards_env, dones, truncated, infos = env.step(all_actions)

        # === 추가 보상: 탱크 따라가기 (공격 보상/패널티 포함) ===
        follow_rewards, prev_distances = calculate_follow_reward(
            env, npc_ids, tank_id, prev_distances,
            actions=all_actions, rewards_env=rewards_env
        )

        # 현재 평균 거리 기록
        tank = env.units.get(tank_id)
        if tank and tank.is_alive:
            step_distances = []
            for aid in npc_ids:
                npc = env.units.get(aid)
                if npc and npc.is_alive:
                    step_distances.append(npc.distance_to(tank))
            if step_distances:
                tank_distances.append(np.mean(step_distances))

        # === NPC 경험 저장 ===
        for agent_id in npc_ids:
            total_reward = rewards_env[agent_id] + follow_rewards.get(agent_id, 0.0)

            transitions_npc.append({
                'agent_id': agent_id,
                'obs': obs_npc_extended[agent_id],
                'action': all_actions[agent_id],
                'reward': total_reward,
                'value': values_npc[agent_id],
                'log_prob': log_probs_npc[agent_id],
                'done': dones[agent_id]
            })
            episode_reward[agent_id] += total_reward

        episode_reward[tank_id] += rewards_env[tank_id]

        obs = next_obs
        episode_length += 1

        if all(dones.values()):
            break

    avg_reward = np.mean([episode_reward[aid] for aid in npc_ids])
    avg_tank_dist = np.mean(tank_distances) if tank_distances else 0

    return {
        'transitions_npc': transitions_npc,
        'episode_length': episode_length,
        'steps': episode_length * 4,
        'winner': infos.get('winner'),
        'avg_reward': avg_reward,
        'avg_tank_dist': avg_tank_dist,
        'policy_type': tank_policy.type_name
    }


def train(args):
    """v11 학습: 공격 보상 강화 + 따라가기 완화"""
    global LOG_FILE, CSV_FILE
    LOG_FILE = args.log_file
    CSV_FILE = args.csv_file

    if LOG_FILE:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== Coop Training v11 - Balanced Attack & Follow ===\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    if CSV_FILE:
        init_csv(CSV_FILE)

    log("=" * 60)
    log("5vs5 Multi-Agent Battle - Coop v11")
    log("=" * 60)
    log("=== v11 핵심 변경 (v10 대비) ===")
    log("  [1] 거리 비례 공격 보상 (탱크 근처일수록 높음!):")
    log("      - 0칸: +15, 2칸: +10, 4칸: +5")
    log("      - 5칸+: -5 강한 패널티 (탱크 무시 방지!)")
    log("  [2] 따라가기 스케일 완화:")
    log("      - scale: 0.5 → 0.3 (싸울 여유 확보)")
    log("      - 접근 보상: 1.0 → 0.8")
    log("      - 이탈 패널티: 1.5 → 1.0")
    log("  [3] 정책 비율 조정:")
    log(f"      - Goal: {args.policy_goal*100:.0f}% (v10: 35%)")
    log(f"      - Model: {args.policy_model*100:.0f}% (v10: 35%)")
    log(f"      - Cond: {args.policy_cond*100:.0f}% (v10: 30%)")
    log("=" * 60)

    # 환경 설정
    env_config = EnvConfig()
    env_config = apply_v11_config(env_config)

    train_config = TrainConfig(
        learning_rate=args.lr,
        total_timesteps=args.total_steps,
        num_steps=args.rollout_steps,
        batch_size=args.batch_size,
    )

    # 환경 생성
    envs = [MultiAgentBattleEnv(EnvConfig()) for _ in range(args.num_envs)]
    for env in envs:
        apply_v11_config(env.config)

    obs = envs[0].reset()
    obs_size_base = list(obs.values())[0].shape[0]  # 229
    obs_size_npc = obs_size_base + 6  # 235
    action_size = envs[0].action_space_size

    log(f"\n=== Obs 차원 정보 ===")
    log(f"  Base obs (env): {obs_size_base}")
    log(f"  Tank info: 6")
    log(f"  NPC obs (학습): {obs_size_npc}")
    log(f"  Action size: {action_size}")
    log(f"  Num envs: {args.num_envs}")

    # === 에이전트 생성 ===
    agent_npc = PPOAgent(obs_size_npc, action_size, train_config)
    agent_b = PPOAgent(obs_size_base, action_size, train_config)

    # === Mixed Tank Policy 생성 ===
    policies = []
    weights = []

    # Type A: Goal-based (순수 따라가기) - 비율 증가!
    if args.policy_goal > 0:
        policies.append(GoalTankPolicy(seed=42))
        weights.append(args.policy_goal)

    # Type B: Model-based (v11) - 비율 감소!
    if args.policy_model > 0 and args.opponent_model and os.path.exists(args.opponent_model):
        policies.append(ModelTankPolicy(args.opponent_model, obs_size_base, action_size))
        weights.append(args.policy_model)

    # Type C: Conditional
    if args.policy_cond > 0:
        policies.append(ConditionalTankPolicy(seed=123, engage_range=args.engage_range))
        weights.append(args.policy_cond)

    if not policies:
        log("ERROR: No tank policies available!")
        return

    tank_policies = [MixedTankPolicy(policies, weights, seed=i) for i in range(args.num_envs)]

    log(f"\n=== Tank Policies ===")
    log(f"  Available: {[p.type_name for p in policies]}")
    log(f"  Weights: {weights}")

    # NPC 모델 로드
    if args.load_model and os.path.exists(args.load_model):
        agent_npc.load(args.load_model)
        log(f"\nNPC model loaded: {args.load_model}")
    else:
        log(f"\nNPC model: Fresh start (새로 학습)")

    # B팀 모델 로드
    if args.opponent_model and os.path.exists(args.opponent_model):
        agent_b.load(args.opponent_model)
        log(f"B Team loaded: {args.opponent_model}")
    else:
        log("WARNING: B Team model not found!")

    # 통계
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    tank_distances = deque(maxlen=100)
    win_counts = {'team_a': 0, 'team_b': 0, 'draw': 0}
    policy_counts = {'Goal': 0, 'Model': 0, 'Conditional': 0}

    total_steps = 0
    episode = 0
    start_time = time.time()

    os.makedirs(args.save_dir, exist_ok=True)
    agent_npc.save(os.path.join(args.save_dir, "model_npc_latest.pt"))

    log("\nTraining started (v11: Balanced Attack & Follow)...")

    while total_steps < train_config.total_timesteps:
        for env_idx, env in enumerate(envs):
            result = run_episode(env, agent_npc, agent_b, tank_policies[env_idx], args)

            for t in result['transitions_npc']:
                agent_npc.store_transition(
                    t['agent_id'], t['obs'], t['action'],
                    t['reward'], t['value'], t['log_prob'], t['done']
                )

            total_steps += result['steps']
            episode += 1
            episode_rewards.append(result['avg_reward'])
            episode_lengths.append(result['episode_length'])
            tank_distances.append(result['avg_tank_dist'])

            # 정책 타입 카운트
            policy_type = result['policy_type']
            if 'Goal' in policy_type:
                policy_counts['Goal'] += 1
            elif 'Model' in policy_type:
                policy_counts['Model'] += 1
            elif 'Conditional' in policy_type:
                policy_counts['Conditional'] += 1

            winner = result['winner']
            if winner == 0:
                win_counts['team_a'] += 1
            elif winner == 1:
                win_counts['team_b'] += 1
            else:
                win_counts['draw'] += 1

        # NPC 학습
        agent_npc.learn()

        # 로깅
        if episode % args.log_interval == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            avg_tank_dist = np.mean(tank_distances) if tank_distances else 0
            win_rate = 100 * win_counts['team_a'] / episode

            log(f"\n[Ep {episode}] Steps: {total_steps:,}")
            log(f"  Reward: {avg_reward:.2f} | Length: {avg_length:.1f}")
            log(f"  Tank Dist: {avg_tank_dist:.2f} | Win A: {win_rate:.1f}%")
            log(f"  Policies: Goal={policy_counts['Goal']}, Model={policy_counts['Model']}, Cond={policy_counts['Conditional']}")
            log(f"  FPS: {fps:.0f}")

            csv_log(episode, total_steps, avg_reward, avg_length,
                   win_counts, fps, avg_tank_dist,
                   f"G{policy_counts['Goal']}_M{policy_counts['Model']}_C{policy_counts['Conditional']}")

        # 저장
        if episode % args.save_interval == 0:
            agent_npc.save(os.path.join(args.save_dir, f"model_npc_ep{episode}.pt"))
            agent_npc.save(os.path.join(args.save_dir, "model_npc_latest.pt"))
            log(f"  Saved: ep{episode}")

    # 최종 저장
    agent_npc.save(os.path.join(args.save_dir, "model_npc_final.pt"))

    log("\n" + "=" * 60)
    log("Training Complete!")
    log(f"Episodes: {episode} | Steps: {total_steps:,}")
    log(f"Final Avg Tank Distance: {np.mean(tank_distances):.2f}")
    log(f"Win Rate A: {100*win_counts['team_a']/episode:.1f}%")
    log(f"Policy Distribution: Goal={policy_counts['Goal']}, Model={policy_counts['Model']}, Cond={policy_counts['Conditional']}")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Coop v11 - Balanced Attack & Follow')
    parser.add_argument('--total-steps', type=int, default=10000000)
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--save-dir', type=str, default='models_coop_v11')
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--load-model', type=str, default=None,
                        help='NPC starting model (None = fresh start)')
    parser.add_argument('--opponent-model', type=str,
                        default='models_v11_10k_episodes/model_final.pt',
                        help='B Team + Tank model path (v11)')

    # === Tank Policy 비율 설정 (v11 조정!) ===
    parser.add_argument('--policy-goal', type=float, default=0.50,
                        help='Type A: Goal-based (v10: 35% → v11: 50%)')
    parser.add_argument('--policy-model', type=float, default=0.15,
                        help='Type B: v11 Model (v10: 35% → v11: 15%)')
    parser.add_argument('--policy-cond', type=float, default=0.35,
                        help='Type C: Conditional (v10: 30% → v11: 35%)')
    parser.add_argument('--engage-range', type=int, default=4,
                        help='Conditional policy 전투 전환 거리')

    parser.add_argument('--log-file', type=str, default='training_log_v11.txt')
    parser.add_argument('--csv-file', type=str, default='training_data_v11.csv')

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
