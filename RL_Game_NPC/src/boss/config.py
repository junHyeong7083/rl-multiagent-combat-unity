"""Boss Raid 환경 설정 및 열거형 (유클리드 연속 공간)"""
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
    """보스 패턴 ID (9개)"""
    SLASH = 0
    CHARGE = 1
    ERUPTION = 2
    TAIL_SWIPE = 3
    MARK = 4
    STAGGER = 5
    CROSS_INFERNO = 6
    CURSED_CHAIN = 7
    SEAL_BREAK = 8          # 봉인 해제: 4장판 중 3개(NPC)를 각각 점유


class PhaseID(IntEnum):
    P1 = 0
    P2 = 1
    P3 = 2


class BossActionID(IntEnum):
    """NPC 행동 공간 (17개 — 8방향 이동)"""
    STAY = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    MOVE_UP_LEFT = 5
    MOVE_UP_RIGHT = 6
    MOVE_DOWN_LEFT = 7
    MOVE_DOWN_RIGHT = 8
    ATTACK_BASIC = 9
    ATTACK_SKILL = 10
    TAUNT = 11
    GUARD = 12
    HEAL = 13
    CLEANSE = 14
    BUFF_ATK = 15
    BUFF_SHIELD = 16


@dataclass
class RoleStatsBoss:
    hp: int
    mp: int
    attack: int
    defense: int
    attack_range: float
    move_speed: float = 1.0
    radius: float = 0.3


# 파티 유닛 스탯 (유클리드)
# v33 플레이타임 연장: HP × 2 (data·attack 고정). 즉사 이벤트(스태거 fail 120, Mark fail 80) 버퍼 확보.
# 힐러 280 HP → 스태거 fail 1회 시 165 HP 생존, 연쇄 wipe 방지.
ROLE_STATS_BOSS: Dict[PartyRole, RoleStatsBoss] = {
    PartyRole.DEALER:  RoleStatsBoss(hp=340, mp=80,  attack=42, defense=5,  attack_range=1.2, move_speed=1.0, radius=0.3),
    PartyRole.TANK:    RoleStatsBoss(hp=680, mp=60,  attack=18, defense=22, attack_range=1.2, move_speed=0.9, radius=0.5),
    PartyRole.HEALER:  RoleStatsBoss(hp=280, mp=150, attack=12, defense=5,  attack_range=3.5, move_speed=1.0, radius=0.3),
    PartyRole.SUPPORT: RoleStatsBoss(hp=360, mp=120, attack=24, defense=10, attack_range=2.5, move_speed=1.0, radius=0.4),
}


@dataclass
class BossConfig:
    # --- 맵 (연속 공간) ---
    map_width: float = 20.0
    map_height: float = 20.0

    # --- 파티 ---
    party_roles: List[PartyRole] = field(default_factory=lambda: [
        PartyRole.DEALER, PartyRole.TANK, PartyRole.HEALER, PartyRole.SUPPORT
    ])
    player_slot: int = 0

    # --- 보스 ---
    # 학습용 기본값은 "승리 가능한" 선에서 설정. 실전(사용자 평가) 때는 스케일업 옵션.
    boss_max_hp: int = 1800         # v33 플레이타임 연장 (파티 HP 2x 와 보조 맞춤, 데미지 값은 고정)
    boss_base_attack: int = 30
    boss_defense: int = 3           # 5 → 3 (파티 딜 관통 잘 되게)
    boss_radius: float = 1.0
    boss_move_speed: float = 0.9
    boss_attack_range: float = 1.8

    # 어그로
    aggro_decay: float = 0.95
    aggro_damage_weight: float = 1.0
    aggro_taunt_bonus: float = 200.0
    aggro_basic_target_cost: float = 30.0

    # 스태거 (P2+) — v27: 달성 가능하도록 재조정
    # 구 설정 300/10/20/30 은 이론 최대치 230 → 100% 실패. 200으로 낮추고 기본 공격 기여 ↑
    stagger_gauge: float = 200.0
    stagger_window_turns: int = 4
    stagger_contrib_basic: float = 12.0
    stagger_contrib_skill: float = 25.0
    stagger_contrib_taunt: float = 35.0
    stagger_fail_damage: int = 120
    stagger_success_grog_turns: int = 2

    # --- 페이즈 ---
    phase_hp_thresholds: Tuple[float, float] = (0.70, 0.35)
    phase_transition_invuln_turns: int = 2

    # --- 동시 활성 텔레그래프 제한 ---
    max_active_telegraphs: int = 2

    # --- 에피소드 ---
    # v35: HP ×2 + 타임아웃 상한 충분히 → max_steps 1000
    max_steps: int = 1000

    # --- 실전 모드 (user test, 시연용) ---
    # 활성 시: 보스 고정 spawn, 인식 범위 안 들어와야 전투 시작, max_steps 무시 (보스 처치까지 진행)
    user_test_mode: bool = False
    boss_spawn_x: float = 17.0           # 맵 우상단 코너 부근
    boss_spawn_y: float = 17.0
    boss_detection_range: float = 5.0    # 보스 인식 범위 (이 안에 플레이어 들어오면 전투 시작)

    # --- 관찰 (B안 재설계) ---
    # Self(15) + Allies(24) + Boss(10) + PatternChannels(9×5=45) + Danger(8) + Escape(4) + Coop(13) + Player(4) = 123
    # 협동 패턴 학습을 위해: 패턴 슬롯 제거→ID별 고정 채널, per-pattern primary target 위치, SEAL spot assignment 포함
    obs_size: int = 123

    # --- 행동 ---
    num_actions: int = 17              # 13 → 17 (대각 4방향 추가)

    # --- 스킬 쿨타임 ---
    skill_cooldown: int = 5

    # --- 패턴별 쿨타임 배율 (페이즈별) ---
    phase_cooldown_scale: Tuple[float, float, float] = (1.0, 0.8, 0.7)

    # --- 페이즈별 패턴 가중치 ---
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

    # --- 패턴 기하 파라미터 ---
    pat_slash_range: float = 3.5        # 부채꼴 반경
    pat_slash_angle_deg: float = 90.0   # 부채꼴 각도
    pat_charge_width: float = 1.0       # 빔 반폭
    pat_eruption_radius: float = 1.6    # 낙뢰 원 반경
    pat_eruption_count: int = 3
    pat_tail_range: float = 5.5
    pat_tail_angle_deg: float = 70.0
    pat_mark_blast_radius: float = 2.8  # 표식 폭발 반경
    pat_mark_escape_distance: float = 5.0
    pat_cross_band_half_width: float = 1.5  # 십자 반폭
    pat_chain_max_distance: float = 3.0

    # Seal Break (봉인 해제)
    pat_seal_spot_distance: float = 3.5     # 보스 중심에서 장판 거리
    pat_seal_spot_radius: float = 1.0       # 장판 반경 (서있으면 판정)
    pat_seal_arrive_turns: int = 20          # 도착 제한 턴 (넉넉하게)
    pat_seal_hold_turns: int = 15           # 유지 필요 턴 (15틱 = 4.5초)
    pat_seal_fail_damage: int = 150          # 실패 = 큰 피해 (즉사 아님 — 학습 기회 보존)
    pat_seal_success_grog: int = 3          # 성공 시 보스 그로기 턴

    # --- 유클리드 "충돌" 기반 기믹 거리 ---
    stagger_gather_radius: float = 3.0
    chain_damage: int = 25

    # ─── 보상 설계 철학 ───
    # 1. 생존 보상은 "최소" (도망 유혹 차단)
    # 2. 딜·힐·파훼는 "큰" 보상 (적극 참여 유도)
    # 3. 시간 패널티 (장기전 억제)
    # 4. 비참여(보스에서 너무 멀리) 패널티

    # --- 공통 ---
    # [v14 재설계] 역할 분화 강화 — 힐러/서포터의 "attack-free-rider" 방지
    rw_boss_damage_per_hp: float = 0.20       # 딜 담당 역할(탱커/서포터) 위주로 보상
    rw_team_damage_per_hp: float = 0.01       # 0.05 → 0.01 (free rider 차단)
    rw_boss_kill: float = 500.0
    rw_phase_clear: float = 25.0              # 15 → 25
    rw_player_alive_step: float = 0.02        # 0.1 → 0.02 (도망 유혹 제거)
    rw_player_death: float = -50.0
    rw_npc_death: float = -20.0               # -10 → -20 (죽으면 더 아픔)
    rw_wipe: float = -150.0                   # -100 → -150
    rw_mechanic_success: float = 8.0          # 5 → 8
    rw_mechanic_fail: float = -15.0           # -10 → -15
    rw_danger_hit: float = -2.0               # -1 → -2
    rw_telegraph_dodge: float = 0.5
    rw_invalid_action: float = -0.3           # -0.5 → -0.3 (탐색 방해 완화)
    rw_time_penalty: float = -0.02            # NEW: 매 턴 시간 패널티
    rw_engage_bonus: float = 0.1              # NEW: 보스 사거리+여유 내에 있으면 보너스
    rw_disengage_penalty: float = -0.15       # NEW: 보스에서 너무 멀리(>8m) 있으면 패널티
    engage_distance: float = 5.0              # 이 거리 이내면 "참여 중"
    disengage_distance: float = 9.0           # 이 거리 초과면 "도망 중"

    # --- 탱커 ---
    rw_tank_aggro_hold: float = 0.5           # 0.3 → 0.5 (탱커의 핵심 역할)
    rw_tank_aggro_lose: float = -1.5          # -1 → -1.5
    rw_tank_taunt_good: float = 1.5           # 1 → 1.5
    rw_tank_close_boss: float = 0.2           # 0.1 → 0.2
    rw_tank_guard_player: float = 0.2         # 0.1 → 0.2

    # --- 힐러 (v14 대폭 강화) ---
    rw_heal_per_hp: float = 0.6               # 0.15 → 0.6 (4배)
    rw_heal_critical: float = 3.0             # 1.0 → 3.0 (위급 힐 매우 큰 보상)
    rw_healer_stagger_atk: float = 0.4        # 유지 (스태거 중에만 딜 유도)
    rw_healer_central: float = 0.15           # 0.08 → 0.15 (중앙 위치 강화)
    rw_healer_ally_low_hp_nearby: float = 0.5 # NEW: 저HP 아군 사거리 내 있으면 보상
    rw_ally_died_nearby_penalty: float = -2.0 # NEW: 힐 가능 거리에서 아군 사망 시 패널티

    # --- 서포터 ---
    rw_buff_hit: float = 0.4                  # 0.3 → 0.4
    rw_buff_tank_pre: float = 0.8             # 0.5 → 0.8
    rw_buff_dealer_grog: float = 1.0          # 0.5 → 1.0 (딜 타임 버프 크게)
    rw_support_stagger_atk: float = 0.4

    # --- 패턴별 협동 (v14 대폭 강화 — 기믹 파훼 학습 유도) ---
    rw_mark_carrier_spread: float = 4.0       # 1.5 → 4.0 (표식 들고 도망)
    rw_mark_other_spread: float = 4.0         # 1.5 → 4.0 (표식자 반대로)
    rw_stagger_gather: float = 1.5            # 0.8 → 1.5
    rw_stagger_success: float = 40.0          # 10 → 40 (스태거 성공 = 큰 보상)
    rw_stagger_contribution: float = 0.3      # NEW: 스태거 게이지 기여 1당 보상
    rw_cross_gather: float = 8.0              # 5 → 8
    rw_cross_split: float = -5.0              # -3 → -5
    rw_chain_hold: float = 0.8                # 0.5 → 0.8
    rw_chain_respect: float = 0.25            # 0.15 → 0.25
    rw_mechanic_success_bonus: float = 15.0   # NEW: 기믹 성공 추가 팀 보상 (전원)
