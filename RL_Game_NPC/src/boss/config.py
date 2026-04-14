"""Boss Raid 환경 설정 및 열거형"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Tuple


class PartyRole(IntEnum):
    """파티 역할 (4인 파티)"""
    DEALER = 0   # 플레이어 고정
    TANK = 1
    HEALER = 2
    SUPPORT = 3


class PatternID(IntEnum):
    """보스 패턴 ID (8개)"""
    SLASH = 0           # P1: 기본 베기 (어그로 1위 근접)
    CHARGE = 1          # P2: 돌진 (랜덤 직선)
    ERUPTION = 2        # P3: 바닥 장판 3개
    TAIL_SWIPE = 3      # P4: 꼬리 휩쓸기 (후방 광역)
    MARK = 4            # P5: 집중 표식 (산개)
    STAGGER = 5         # P6: 스태거 체크
    CROSS_INFERNO = 6   # P7: 십자 화염 (전멸기)
    CURSED_CHAIN = 7    # P8: 저주 연결 (2명 묶기)


class PhaseID(IntEnum):
    """보스 페이즈"""
    P1 = 0
    P2 = 1
    P3 = 2


class BossActionID(IntEnum):
    """NPC 행동 공간 (13개)"""
    STAY = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ATTACK_BASIC = 5       # 근접 기본 공격
    ATTACK_SKILL = 6       # 스킬 공격 (전원 사용 가능, 쿨 있음)
    TAUNT = 7              # 도발 (탱커 전용)
    GUARD = 8              # 방어 자세 (탱커 전용)
    HEAL = 9               # 치유 (힐러 전용)
    CLEANSE = 10           # 해제 (힐러 전용)
    BUFF_ATK = 11          # 공격 버프 (서포터 전용)
    BUFF_SHIELD = 12       # 방어 버프 (서포터 전용)


@dataclass
class RoleStatsBoss:
    hp: int
    mp: int
    attack: int
    defense: int
    attack_range: int
    move_speed: int = 1


# 파티 유닛 스탯 (보스전 전용, 원 논문보다 전반적 상향)
ROLE_STATS_BOSS: Dict[PartyRole, RoleStatsBoss] = {
    PartyRole.DEALER:  RoleStatsBoss(hp=120, mp=80,  attack=35, defense=5,  attack_range=1),
    PartyRole.TANK:    RoleStatsBoss(hp=250, mp=60,  attack=15, defense=20, attack_range=1),
    PartyRole.HEALER:  RoleStatsBoss(hp=100, mp=150, attack=10, defense=5,  attack_range=3),
    PartyRole.SUPPORT: RoleStatsBoss(hp=130, mp=120, attack=20, defense=10, attack_range=2),
}


@dataclass
class BossConfig:
    # --- 맵 ---
    map_width: int = 20
    map_height: int = 20

    # --- 파티 ---
    party_roles: List[PartyRole] = field(default_factory=lambda: [
        PartyRole.DEALER, PartyRole.TANK, PartyRole.HEALER, PartyRole.SUPPORT
    ])
    player_slot: int = 0  # DEALER 슬롯이 플레이어

    # --- 보스 ---
    boss_max_hp: int = 5000
    boss_base_attack: int = 30
    boss_defense: int = 5
    boss_size: int = 2          # 2x2 점유
    boss_move_speed: int = 1
    boss_attack_range: int = 1  # 기본 근접

    # 어그로
    aggro_decay: float = 0.95
    aggro_damage_weight: float = 1.0
    aggro_taunt_bonus: float = 200.0
    aggro_basic_target_cost: float = 30.0  # 기본공격 시 해당 유닛 어그로 소비

    # 스태거 (P2+)
    stagger_gauge: float = 300.0
    stagger_window_turns: int = 4
    stagger_contrib_basic: float = 10.0
    stagger_contrib_skill: float = 20.0
    stagger_contrib_taunt: float = 30.0
    stagger_fail_damage: int = 120
    stagger_success_grog_turns: int = 2

    # --- 페이즈 ---
    phase_hp_thresholds: Tuple[float, float] = (0.70, 0.35)  # 진입 임계값
    phase_transition_invuln_turns: int = 2

    # --- 동시 활성 텔레그래프 제한 ---
    max_active_telegraphs: int = 2

    # --- 에피소드 ---
    max_steps: int = 200

    # --- 관찰 ---
    obs_size: int = 92  # 설계 문서 참조

    # --- 행동 ---
    num_actions: int = 13

    # --- 스킬 쿨타임 (공통 5턴) ---
    skill_cooldown: int = 5

    # --- 패턴별 쿨타임 배율 (페이즈별) ---
    phase_cooldown_scale: Tuple[float, float, float] = (1.0, 0.8, 0.7)

    # --- 페이즈별 패턴 가중치 ---
    # dict[PhaseID][PatternID] = weight (0이면 미등장)
    pattern_weights: Dict[int, Dict[int, float]] = field(default_factory=lambda: {
        int(PhaseID.P1): {
            int(PatternID.SLASH): 0.30, int(PatternID.CHARGE): 0.25,
            int(PatternID.ERUPTION): 0.25, int(PatternID.TAIL_SWIPE): 0.20,
        },
        int(PhaseID.P2): {
            int(PatternID.SLASH): 0.20, int(PatternID.CHARGE): 0.15,
            int(PatternID.ERUPTION): 0.20, int(PatternID.TAIL_SWIPE): 0.15,
            int(PatternID.MARK): 0.15, int(PatternID.STAGGER): 0.15,
        },
        int(PhaseID.P3): {
            int(PatternID.SLASH): 0.15, int(PatternID.CHARGE): 0.10,
            int(PatternID.ERUPTION): 0.15, int(PatternID.TAIL_SWIPE): 0.10,
            int(PatternID.MARK): 0.15, int(PatternID.STAGGER): 0.15,
            int(PatternID.CROSS_INFERNO): 0.10, int(PatternID.CURSED_CHAIN): 0.10,
        },
    })

    # --- 보상 가중치 ---
    # 공통
    rw_boss_damage_per_hp: float = 0.05
    rw_boss_kill: float = 100.0
    rw_phase_clear: float = 15.0
    rw_player_alive_step: float = 0.1
    rw_player_death: float = -50.0
    rw_npc_death: float = -10.0
    rw_wipe: float = -100.0
    rw_mechanic_success: float = 5.0
    rw_mechanic_fail: float = -10.0
    rw_danger_hit: float = -1.0
    rw_telegraph_dodge: float = 0.5
    rw_invalid_action: float = -0.5

    # 탱커
    rw_tank_aggro_hold: float = 0.3
    rw_tank_aggro_lose: float = -1.0
    rw_tank_taunt_good: float = 1.0
    rw_tank_close_boss: float = 0.1
    rw_tank_guard_player: float = 0.1

    # 힐러
    rw_heal_per_hp: float = 0.1
    rw_heal_critical: float = 0.5
    rw_healer_stagger_atk: float = 0.3
    rw_healer_central: float = 0.05

    # 서포터
    rw_buff_hit: float = 0.3
    rw_buff_tank_pre: float = 0.5
    rw_buff_dealer_grog: float = 0.5
    rw_support_stagger_atk: float = 0.3

    # 패턴별 특수 보상
    rw_mark_carrier_spread: float = 1.0
    rw_mark_other_spread: float = 1.0
    rw_stagger_gather: float = 0.5
    rw_stagger_success: float = 5.0
    rw_cross_gather: float = 3.0
    rw_cross_split: float = -2.0
    rw_chain_hold: float = 0.3
    rw_chain_respect: float = 0.1
