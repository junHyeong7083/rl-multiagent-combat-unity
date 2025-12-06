"""유닛(Unit) 클래스 구현"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

from .config import RoleType, RoleStats, ROLE_STATS, ActionType


@dataclass
class Unit:
    """게임 유닛 클래스"""
    unit_id: int
    team_id: int  # 0: Team A, 1: Team B
    role: RoleType
    x: int = 0
    y: int = 0
    stat_multiplier: float = 1.0  # 스탯 배율 (비대칭 학습용)

    # 스탯 (초기화 시 역할에 따라 설정)
    max_hp: int = field(init=False)
    hp: int = field(init=False)
    max_mp: int = field(init=False)
    mp: int = field(init=False)
    attack: int = field(init=False)
    defense: int = field(init=False)
    attack_range: int = field(init=False)
    move_speed: int = field(init=False)

    # 쿨타임
    skill_cooldowns: dict = field(default_factory=dict)

    # 상태
    is_alive: bool = True
    buffs: dict = field(default_factory=dict)
    total_damage_taken: int = 0  # 누적 흡수 데미지 (탱커 보상용)

    # 도발 상태 (탱커에게 도발당한 경우)
    taunted_by: Optional[str] = None  # 도발한 탱커의 agent_id
    taunt_duration: int = 0  # 도발 남은 턴

    # 버프 추적 (서포터가 누구에게 버프를 줬는지)
    buffed_by: Optional[str] = None  # 버프를 준 서포터의 agent_id

    def __post_init__(self):
        """역할에 따른 스탯 초기화 (배율 적용)"""
        stats = ROLE_STATS[self.role]
        m = self.stat_multiplier
        self.max_hp = int(stats.hp * m)
        self.hp = self.max_hp
        self.max_mp = int(stats.mp * m)
        self.mp = self.max_mp
        self.attack = int(stats.attack * m)
        self.defense = int(stats.defense * m)
        self.attack_range = stats.attack_range  # 사거리는 배율 미적용
        self.move_speed = stats.move_speed  # 이속도 배율 미적용

        # 스킬 쿨타임 초기화 (역할별 스킬만 해당)
        self.skill_cooldowns = {
            ActionType.SKILL_AOE: 0,     # 딜러
            ActionType.SKILL_HEAL: 0,    # 힐러
            ActionType.SKILL_TAUNT: 0,   # 탱커
            ActionType.SKILL_PIERCE: 0,  # 레인저
            ActionType.SKILL_BUFF: 0,    # 서포터
        }

        # 도발/버프 상태 초기화
        self.taunted_by = None
        self.taunt_duration = 0
        self.buffed_by = None

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @position.setter
    def position(self, pos: Tuple[int, int]):
        self.x, self.y = pos

    def take_damage(self, damage: int) -> int:
        """데미지를 받음. 실제 받은 데미지 반환"""
        # 방어력 적용 (최소 1 데미지)
        actual_damage = max(1, damage - self.defense)
        self.hp -= actual_damage
        self.total_damage_taken += actual_damage  # 누적 데미지 추적

        if self.hp <= 0:
            self.hp = 0
            self.is_alive = False

        return actual_damage

    def heal(self, amount: int) -> int:
        """회복. 실제 회복량 반환"""
        if not self.is_alive:
            return 0

        old_hp = self.hp
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp - old_hp

    def use_mp(self, cost: int) -> bool:
        """MP 소모. 성공 여부 반환"""
        if self.mp >= cost:
            self.mp -= cost
            return True
        return False

    def recover_mp(self, amount: int) -> int:
        """MP 회복. 실제 회복량 반환"""
        old_mp = self.mp
        self.mp = min(self.max_mp, self.mp + amount)
        return self.mp - old_mp

    def can_attack(self, target: 'Unit') -> bool:
        """타겟을 공격할 수 있는지 확인"""
        if not self.is_alive or not target.is_alive:
            return False
        if self.team_id == target.team_id:
            return False

        distance = self.distance_to(target)
        return distance <= self.attack_range

    def distance_to(self, other: 'Unit') -> float:
        """다른 유닛과의 맨해튼 거리"""
        return abs(self.x - other.x) + abs(self.y - other.y)

    def distance_to_pos(self, pos: Tuple[int, int]) -> float:
        """특정 위치까지의 맨해튼 거리"""
        return abs(self.x - pos[0]) + abs(self.y - pos[1])

    def get_attack_damage(self) -> int:
        """현재 공격력 반환 (버프 적용)"""
        buff_data = self.buffs.get('attack')
        bonus = buff_data['value'] if buff_data else 0
        return self.attack + bonus

    def update_cooldowns(self):
        """쿨타임 감소 (턴 종료 시 호출)"""
        for skill in self.skill_cooldowns:
            if self.skill_cooldowns[skill] > 0:
                self.skill_cooldowns[skill] -= 1

    def can_use_skill(self, skill: ActionType) -> bool:
        """스킬 사용 가능 여부"""
        return self.skill_cooldowns.get(skill, 0) == 0

    def set_cooldown(self, skill: ActionType, turns: int):
        """스킬 쿨타임 설정"""
        self.skill_cooldowns[skill] = turns

    def apply_buff(self, buff_type: str, value: int, duration: int):
        """버프 적용"""
        self.buffs[buff_type] = {
            'value': value,
            'duration': duration
        }

    def update_buffs(self):
        """버프 지속시간 감소"""
        to_remove = []
        for buff_type, buff_data in self.buffs.items():
            buff_data['duration'] -= 1
            if buff_data['duration'] <= 0:
                to_remove.append(buff_type)

        for buff_type in to_remove:
            del self.buffs[buff_type]
            # 공격력 버프가 끝나면 buffed_by 초기화
            if buff_type == 'attack':
                self.buffed_by = None

    def update_taunt(self):
        """도발 상태 업데이트 (턴 종료 시 호출)"""
        if self.taunt_duration > 0:
            self.taunt_duration -= 1
            if self.taunt_duration <= 0:
                self.taunted_by = None

    def apply_taunt(self, taunter_agent_id: str, duration: int):
        """도발 적용"""
        self.taunted_by = taunter_agent_id
        self.taunt_duration = duration

    def is_taunted(self) -> bool:
        """도발 상태인지 확인"""
        return self.taunted_by is not None and self.taunt_duration > 0

    def apply_buff_from(self, supporter_agent_id: str, buff_type: str, value: int, duration: int):
        """서포터로부터 버프 적용 (추적 포함)"""
        self.apply_buff(buff_type, value, duration)
        self.buffed_by = supporter_agent_id

    def get_state_vector(self) -> np.ndarray:
        """유닛 상태를 벡터로 반환"""
        # [hp_ratio, mp_ratio, x_norm, y_norm, role_onehot(5), is_alive]
        role_onehot = np.zeros(5)
        role_onehot[self.role] = 1

        state = np.array([
            self.hp / self.max_hp,
            self.mp / self.max_mp,
            self.x / 20.0,  # 정규화 (맵 크기 기준)
            self.y / 20.0,
            float(self.is_alive),
        ])

        return np.concatenate([state, role_onehot])

    def reset(self, x: int, y: int):
        """유닛 상태 리셋"""
        self.x = x
        self.y = y
        self.hp = self.max_hp
        self.mp = self.max_mp
        self.is_alive = True
        self.buffs = {}
        self.total_damage_taken = 0  # 누적 데미지 초기화

        # 스킬 쿨타임 초기화
        self.skill_cooldowns = {
            ActionType.SKILL_AOE: 0,     # 딜러
            ActionType.SKILL_HEAL: 0,    # 힐러
            ActionType.SKILL_TAUNT: 0,   # 탱커
            ActionType.SKILL_PIERCE: 0,  # 레인저
            ActionType.SKILL_BUFF: 0,    # 서포터
        }

        # 도발/버프 상태 초기화
        self.taunted_by = None
        self.taunt_duration = 0
        self.buffed_by = None

    def __repr__(self):
        team = "A" if self.team_id == 0 else "B"
        return f"Unit({team}{self.unit_id}, {self.role.name}, HP:{self.hp}/{self.max_hp}, pos:({self.x},{self.y}))"
