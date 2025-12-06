"""환경 설정 및 Config 클래스"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import IntEnum


class TileType(IntEnum):
    """타일 종류"""
    EMPTY = 0      # 평지
    WALL = 1       # 벽 (통과 불가)
    DANGER = 2     # 위험 타일
    BUFF_ATK = 3   # 공격력 버프
    BUFF_DEF = 4   # 방어력 버프
    BUFF_HEAL = 5  # 힐 버프


class RoleType(IntEnum):
    """유닛 역할"""
    TANK = 0       # 탱커: 높은 HP, 낮은 공격
    DEALER = 1     # 딜러: 높은 공격, 낮은 HP
    HEALER = 2     # 힐러: 힐 스킬 보유
    RANGER = 3     # 원거리: 긴 공격 범위
    SUPPORT = 4    # 서포터: 버프/디버프


class ActionType(IntEnum):
    """행동 종류"""
    STAY = 0       # 제자리
    MOVE_UP = 1    # 위로 이동
    MOVE_DOWN = 2  # 아래로 이동
    MOVE_LEFT = 3  # 왼쪽으로 이동
    MOVE_RIGHT = 4 # 오른쪽으로 이동
    ATTACK_NEAREST = 5  # 가장 가까운 적 공격
    ATTACK_LOWEST = 6   # HP 가장 낮은 적 공격
    SKILL_AOE = 7       # 범위 공격 스킬 (딜러 전용)
    SKILL_HEAL = 8      # 힐 스킬 (힐러 전용 - 범위 힐)
    SKILL_TAUNT = 9     # 도발 스킬 (탱커 전용)
    SKILL_PIERCE = 10   # 관통샷 스킬 (레인저 전용)
    SKILL_BUFF = 11     # 버프 스킬 (서포터 전용)


@dataclass
class RoleStats:
    """역할별 기본 스탯"""
    hp: int
    mp: int
    attack: int
    defense: int
    attack_range: int
    move_speed: int = 1


# 역할별 기본 스탯 정의
ROLE_STATS: Dict[RoleType, RoleStats] = {
    RoleType.TANK: RoleStats(hp=150, mp=30, attack=10, defense=15, attack_range=1),
    RoleType.DEALER: RoleStats(hp=80, mp=50, attack=25, defense=5, attack_range=1),
    RoleType.HEALER: RoleStats(hp=70, mp=100, attack=8, defense=5, attack_range=2),
    RoleType.RANGER: RoleStats(hp=60, mp=60, attack=20, defense=3, attack_range=4),
    RoleType.SUPPORT: RoleStats(hp=90, mp=80, attack=12, defense=8, attack_range=2),
}


@dataclass
class EnvConfig:
    """환경 설정"""
    # 맵 설정
    map_width: int = 20
    map_height: int = 20
    wall_density: float = 0.1  # 벽 밀도
    danger_density: float = 0.05  # 위험 타일 밀도
    buff_density: float = 0.03  # 버프 타일 밀도

    # 팀 설정
    team_size: int = 5
    num_teams: int = 2

    # 게임 설정
    max_steps: int = 200  # 최대 턴 수
    simultaneous_actions: bool = True  # 동시 행동 적용

    # 관찰 설정
    obs_range: int = 5  # 관찰 범위 (5x5 패치)

    # 행동 설정
    num_actions: int = 12  # 행동 개수 (v11 모델 호환)

    # 보상 설정 (공격적 + 역할별 차별화)
    reward_win: float = 10.0
    reward_lose: float = -10.0
    reward_kill: float = 5.0        # 적 처치 보상
    reward_death: float = -0.3      # 사망 패널티 완화 (공격 유도)
    reward_damage: float = 0.3      # 데미지당 보상 대폭 증가
    reward_heal: float = 0.1        # 힐량당 보상 증가
    reward_danger_tile: float = -0.02  # 위험 타일 패널티
    reward_stay: float = -0.5       # 제자리 패널티 강화 (존버 방지)
    reward_approach: float = 0.3    # 적에게 접근 보상 대폭 증가
    reward_time_penalty: float = -0.05  # 매 턴 패널티 증가 (빠른 교전 유도)
    reward_draw: float = -8.0       # 무승부 패널티 강화 (존버 시 무승부 확률 높음)
    reward_in_combat: float = 0.15  # 전투 범위 내 보상 증가

    # 역할별 추가 보상
    reward_tank_aggro: float = 0.15      # 탱커: 데미지 받으면 보상 (앞에서 어그로)
    reward_tank_death: float = -2.0      # 탱커: 사망 시 추가 패널티
    reward_tank_absorb_bonus: float = 0.02  # 탱커: 흡수 데미지당 사망 패널티 감소
    reward_dealer_damage: float = 0.3    # 딜러: 데미지 주면 추가 보상 (공격 유도)
    reward_dealer_death: float = -1.0    # 딜러: 사망 시 추가 패널티 (화력 손실)
    reward_healer_heal: float = 0.15     # 힐러: 힐하면 추가 보상
    reward_healer_death: float = -1.5    # 힐러: 사망 시 추가 패널티 (지속력 손실)
    reward_ranger_range_atk: float = 0.1 # 레인저: 원거리 공격 보상 (줄임 - 존버 방지)
    reward_support_nearby: float = 0.03  # 서포터: 아군 근처 보상

    # 스킬 설정 (모든 스킬 쿨타임 5스텝)
    skill_cooldown: int = 5  # 공통 쿨타임

    # 딜러: AOE 범위 공격
    skill_aoe_damage: int = 15
    skill_aoe_range: int = 2
    skill_aoe_cost: int = 20

    # 힐러: 범위 힐 (힐러만 사용 가능)
    skill_heal_amount: int = 30
    skill_heal_range: int = 3
    skill_heal_cost: int = 25

    # 탱커: 도발 (범위 내 적 도발, 도발된 적은 탱커 우선 공격)
    skill_taunt_range: int = 3
    skill_taunt_duration: int = 3  # 도발 지속 턴
    skill_taunt_cost: int = 15

    # 레인저: 관통샷 (타겟-레인저 일직선 방향으로 관통)
    skill_pierce_damage: int = 18
    skill_pierce_range: int = 5  # 관통 최대 거리
    skill_pierce_cost: int = 25

    # 서포터: 버프 (아군에게 공격력 버프)
    skill_buff_range: int = 3
    skill_buff_value: int = 10  # 공격력 증가량
    skill_buff_duration: int = 5  # 버프 지속 턴
    skill_buff_cost: int = 20

    # 역할별 스킬 보상
    reward_taunt_per_enemy: float = 1.0   # 도발된 적 1명당 보상
    reward_aoe_per_enemy: float = 0.5     # AOE 적중 적 1명당 보상
    reward_heal_per_ally: float = 0.8     # 범위힐 아군 1명당 보상
    reward_pierce_per_enemy: float = 0.7  # 관통 적중 적 1명당 보상
    reward_buff_ongoing: float = 0.3      # 버프받은 아군 생존 시 매 턴 보상

    # 팀 구성 (역할 배분)
    team_composition: List[RoleType] = field(default_factory=lambda: [
        RoleType.TANK,
        RoleType.DEALER,
        RoleType.HEALER,
        RoleType.RANGER,
        RoleType.SUPPORT
    ])

    # 팀별 스탯 배율 (대칭 학습)
    team_a_stat_multiplier: float = 1.0  # A팀 100%
    team_b_stat_multiplier: float = 1.0  # B팀 100%

    # 플레이어 협동 모드 설정
    player_idx: int = -1    # A팀 플레이어 유닛 인덱스 (-1: 없음, 0~4: 해당 유닛)
    player_idx_b: int = -1  # B팀 플레이어 유닛 인덱스 (-1: 없음, 0~4: 해당 유닛)

    # 협동 보상 설정 (대폭 강화 - 실제 NPC처럼 서포트하도록)
    reward_protect_player: float = 1.5    # 탱커가 플레이어(딜러/힐러) 앞에 있으면 보상
    reward_support_player: float = 1.0    # 힐러가 플레이어 근처에서 힐하면 보상
    reward_follow_tank: float = 0.8       # 딜러/힐러가 탱커 뒤에 있으면 보상
    reward_near_player: float = 0.5       # 플레이어 근처(거리 3 이내)에 있으면 보상
    reward_approach_player: float = 0.2   # 플레이어에게 가까워지면 보상 (거리당)
    coop_distance_threshold: int = 5      # 협동 거리 임계값


@dataclass
class TrainConfig:
    """학습 설정"""
    # PPO 하이퍼파라미터
    learning_rate: float = 3e-4
    gamma: float = 0.99  # 할인율
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip
    entropy_coef: float = 0.05  # 엔트로피 계수 (0.01→0.05: 탐색 증가)
    value_coef: float = 0.5  # 가치 손실 계수
    max_grad_norm: float = 0.5  # 그래디언트 클리핑

    # 학습 설정
    num_envs: int = 8  # 병렬 환경 수
    num_steps: int = 128  # 롤아웃 스텝 수
    num_epochs: int = 4  # 에폭 수
    batch_size: int = 256
    total_timesteps: int = 1_000_000

    # 네트워크 설정
    hidden_size: int = 256
    num_layers: int = 2

    # 저장/로깅
    save_interval: int = 10000
    log_interval: int = 1000


# 시각화 설정
@dataclass
class VisualConfig:
    """시각화 설정"""
    cell_size: int = 32  # 셀 크기 (픽셀)
    fps: int = 10  # 프레임 레이트

    # 색상 (RGB)
    color_empty: Tuple[int, int, int] = (200, 200, 200)
    color_wall: Tuple[int, int, int] = (50, 50, 50)
    color_danger: Tuple[int, int, int] = (255, 100, 100)
    color_buff_atk: Tuple[int, int, int] = (255, 200, 100)
    color_buff_def: Tuple[int, int, int] = (100, 200, 255)
    color_buff_heal: Tuple[int, int, int] = (100, 255, 100)

    color_team_a: Tuple[int, int, int] = (0, 100, 255)
    color_team_b: Tuple[int, int, int] = (255, 50, 50)

    color_hp_bar: Tuple[int, int, int] = (0, 255, 0)
    color_hp_bg: Tuple[int, int, int] = (100, 100, 100)
