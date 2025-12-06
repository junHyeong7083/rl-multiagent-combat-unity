"""MultiAgentBattleEnv 환경 클래스 구현"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .config import (
    EnvConfig, TileType, RoleType, ActionType,
    ROLE_STATS
)
from .unit import Unit
from .game_map import GameMap


class MultiAgentBattleEnv:
    """5vs5 멀티에이전트 전투 환경"""

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()

        # 맵 생성
        self.game_map = GameMap(self.config)

        # 유닛 관리
        self.units: Dict[str, Unit] = {}  # agent_id -> Unit
        self.team_a: List[str] = []
        self.team_b: List[str] = []

        # 게임 상태
        self.current_step = 0
        self.done = False
        self.winner: Optional[int] = None  # 0: Team A, 1: Team B, None: 무승부

        # 턴 내 발생한 이벤트 (보상 계산용)
        self.step_events: Dict[str, List[dict]] = defaultdict(list)

        # 관찰 공간 크기 계산
        self._calculate_obs_size()

        # 초기화
        self.reset()

    def _calculate_obs_size(self):
        """관찰 벡터 크기 계산"""
        # 자기 상태: hp, mp, x, y, alive, role_onehot(5) = 10
        self_state_size = 10

        # 아군 상태 (4명): 각 10 = 40
        ally_state_size = 4 * 10

        # 적군 상태 (5명): 각 10 = 50
        enemy_state_size = 5 * 10

        # 지형 상태: (2*obs_range+1)^2 * one-hot(6) 타일 타입
        # 간략화: (2*obs_range+1)^2 정수값
        obs_range = self.config.obs_range
        terrain_size = (2 * obs_range + 1) ** 2

        # 전역 정보: 현재 턴, 팀 ID = 2
        global_size = 2

        # 플레이어 정보: 항상 포함 (6차원)
        # 협동 모드 아닐 때는 "플레이어 없음" one-hot
        player_info_size = 6

        self.obs_size = self_state_size + ally_state_size + enemy_state_size + terrain_size + global_size + player_info_size

    @property
    def num_agents(self) -> int:
        return self.config.team_size * self.config.num_teams

    @property
    def action_space_size(self) -> int:
        return self.config.num_actions

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """환경 리셋"""
        if seed is not None:
            np.random.seed(seed)

        # 맵 생성
        self.game_map.reset()
        self.game_map.generate(seed)

        # 유닛 초기화
        self.units.clear()
        self.team_a.clear()
        self.team_b.clear()

        # Team A 유닛 생성 (스탯 배율 적용)
        spawn_a = self.game_map.get_spawn_positions(0, self.config.team_size)
        for i, (x, y) in enumerate(spawn_a):
            agent_id = f"team_a_{i}"
            role = self.config.team_composition[i]
            unit = Unit(unit_id=i, team_id=0, role=role, x=x, y=y,
                       stat_multiplier=self.config.team_a_stat_multiplier)
            self.units[agent_id] = unit
            self.team_a.append(agent_id)
            self.game_map.place_unit(agent_id, x, y)

        # Team B 유닛 생성 (스탯 배율 적용)
        spawn_b = self.game_map.get_spawn_positions(1, self.config.team_size)
        for i, (x, y) in enumerate(spawn_b):
            agent_id = f"team_b_{i}"
            role = self.config.team_composition[i]
            unit = Unit(unit_id=i, team_id=1, role=role, x=x, y=y,
                       stat_multiplier=self.config.team_b_stat_multiplier)
            self.units[agent_id] = unit
            self.team_b.append(agent_id)
            self.game_map.place_unit(agent_id, x, y)

        # 게임 상태 초기화
        self.current_step = 0
        self.done = False
        self.winner = None
        self.step_events.clear()

        return self._get_observations()

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # dones
        Dict[str, bool],        # truncated
        Dict[str, Any]          # infos
    ]:
        """한 스텝 실행"""
        if self.done:
            return self._get_observations(), self._get_zero_rewards(), self._get_dones(), self._get_truncated(), {}

        # 이벤트 초기화
        self.step_events.clear()

        # 1. 모든 에이전트의 행동 수집 및 적용
        if self.config.simultaneous_actions:
            self._apply_simultaneous_actions(actions)
        else:
            self._apply_sequential_actions(actions)

        # 2. 지형 효과 적용
        self._apply_terrain_effects()

        # 3. 쿨타임/버프 업데이트
        self._update_unit_states()

        # 4. 위험 타일 업데이트
        self.game_map.update_danger_tiles(self.current_step)

        # 5. 턴 증가
        self.current_step += 1

        # 6. 게임 종료 체크
        self._check_game_end()

        # 7. 보상 계산
        rewards = self._calculate_rewards()

        # 8. 관찰 생성
        observations = self._get_observations()

        # 9. 종료 상태
        dones = self._get_dones()
        truncated = self._get_truncated()

        # 10. 추가 정보
        infos = self._get_infos()

        return observations, rewards, dones, truncated, infos

    def _apply_simultaneous_actions(self, actions: Dict[str, int]):
        """모든 에이전트의 행동을 동시에 적용"""
        # 이동 전 위치 저장 (approach 보상 계산용)
        old_distances = {}
        for agent_id, unit in self.units.items():
            if unit.is_alive:
                enemies = self._get_enemy_units(unit.team_id)
                if enemies:
                    min_dist = min(abs(unit.x - e.x) + abs(unit.y - e.y) for e in enemies)
                    old_distances[agent_id] = min_dist

        # 먼저 이동 처리 (충돌 해결)
        move_intentions = {}
        for agent_id, action in actions.items():
            unit = self.units.get(agent_id)
            if unit and unit.is_alive:
                # STAY 행동에 패널티 이벤트 기록
                if action == ActionType.STAY:
                    self.step_events[agent_id].append({"type": "stay"})
                if ActionType.STAY <= action <= ActionType.MOVE_RIGHT:
                    new_pos = self._get_move_position(unit, action)
                    move_intentions[agent_id] = new_pos

        # 충돌 해결 및 이동 적용
        self._resolve_movements(move_intentions)

        # 이동 후 approach 보상 계산
        for agent_id, unit in self.units.items():
            if unit.is_alive and agent_id in old_distances:
                enemies = self._get_enemy_units(unit.team_id)
                if enemies:
                    new_dist = min(abs(unit.x - e.x) + abs(unit.y - e.y) for e in enemies)
                    if new_dist < old_distances[agent_id]:
                        self.step_events[agent_id].append({"type": "approach"})

        # 공격/스킬 처리
        for agent_id, action in actions.items():
            unit = self.units.get(agent_id)
            if unit and unit.is_alive:
                if action >= ActionType.ATTACK_NEAREST:
                    self._apply_action(agent_id, action)

    def _apply_sequential_actions(self, actions: Dict[str, int]):
        """팀 순서대로 행동 적용 (A팀 -> B팀)"""
        # Team A 먼저
        for agent_id in self.team_a:
            if agent_id in actions:
                unit = self.units[agent_id]
                if unit.is_alive:
                    self._apply_single_action(agent_id, actions[agent_id])

        # Team B
        for agent_id in self.team_b:
            if agent_id in actions:
                unit = self.units[agent_id]
                if unit.is_alive:
                    self._apply_single_action(agent_id, actions[agent_id])

    def _get_move_position(self, unit: Unit, action: int) -> Tuple[int, int]:
        """이동 행동에 따른 새 위치 계산"""
        dx, dy = 0, 0
        if action == ActionType.MOVE_UP:
            dy = -1
        elif action == ActionType.MOVE_DOWN:
            dy = 1
        elif action == ActionType.MOVE_LEFT:
            dx = -1
        elif action == ActionType.MOVE_RIGHT:
            dx = 1

        new_x = unit.x + dx
        new_y = unit.y + dy

        # 이동 가능 여부 확인
        if self.game_map.is_walkable(new_x, new_y):
            return (new_x, new_y)
        return (unit.x, unit.y)  # 이동 불가시 제자리

    def _resolve_movements(self, move_intentions: Dict[str, Tuple[int, int]]):
        """이동 충돌 해결"""
        # 같은 위치로 이동하려는 유닛들 처리
        target_positions = defaultdict(list)
        for agent_id, pos in move_intentions.items():
            target_positions[pos].append(agent_id)

        # 충돌 없는 이동 먼저 처리
        for pos, agents in target_positions.items():
            if len(agents) == 1:
                agent_id = agents[0]
                unit = self.units[agent_id]
                # 목표 위치에 다른 유닛이 없는지 확인
                if not self.game_map.is_occupied(pos[0], pos[1]) or pos == (unit.x, unit.y):
                    self._move_unit(agent_id, pos)
            else:
                # 충돌: 모든 유닛 제자리
                pass

    def _move_unit(self, agent_id: str, new_pos: Tuple[int, int]):
        """유닛 이동 실행"""
        unit = self.units[agent_id]
        old_pos = (unit.x, unit.y)

        if old_pos != new_pos:
            self.game_map.remove_unit(old_pos[0], old_pos[1])
            unit.x, unit.y = new_pos
            self.game_map.place_unit(agent_id, new_pos[0], new_pos[1])

    def _apply_single_action(self, agent_id: str, action: int):
        """단일 행동 적용"""
        unit = self.units[agent_id]

        if action <= ActionType.MOVE_RIGHT:
            # 이동
            new_pos = self._get_move_position(unit, action)
            if self.game_map.can_move_to(new_pos[0], new_pos[1]):
                self._move_unit(agent_id, new_pos)
        else:
            # 공격/스킬
            self._apply_action(agent_id, action)

    def _apply_action(self, agent_id: str, action: int):
        """공격/스킬 행동 적용 (역할별 스킬 제한)"""
        unit = self.units[agent_id]
        enemies = self._get_enemy_units(unit.team_id)
        cooldown = getattr(self.config, 'skill_cooldown', 20)

        if action == ActionType.ATTACK_NEAREST:
            # 가장 가까운 적 공격 (도발 상태면 도발한 탱커 우선)
            target = self._find_attack_target(unit, enemies, prefer_nearest=True)
            if target and unit.can_attack(target):
                self._execute_attack(agent_id, target)

        elif action == ActionType.ATTACK_LOWEST:
            # HP 가장 낮은 적 공격 (도발 상태면 도발한 탱커 우선)
            target = self._find_attack_target(unit, enemies, prefer_nearest=False)
            if target and unit.can_attack(target):
                self._execute_attack(agent_id, target)

        elif action == ActionType.SKILL_AOE:
            # 딜러 전용: 범위 공격
            if unit.role == RoleType.DEALER:
                if unit.can_use_skill(ActionType.SKILL_AOE):
                    if unit.use_mp(self.config.skill_aoe_cost):
                        self._execute_aoe_attack(agent_id)
                        unit.set_cooldown(ActionType.SKILL_AOE, cooldown)

        elif action == ActionType.SKILL_HEAL:
            # 힐러 전용: 범위 힐
            if unit.role == RoleType.HEALER:
                if unit.can_use_skill(ActionType.SKILL_HEAL):
                    if unit.use_mp(self.config.skill_heal_cost):
                        self._execute_aoe_heal(agent_id)
                        unit.set_cooldown(ActionType.SKILL_HEAL, cooldown)

        elif action == ActionType.SKILL_TAUNT:
            # 탱커 전용: 도발
            if unit.role == RoleType.TANK:
                if unit.can_use_skill(ActionType.SKILL_TAUNT):
                    cost = getattr(self.config, 'skill_taunt_cost', 15)
                    if unit.use_mp(cost):
                        self._execute_taunt(agent_id)
                        unit.set_cooldown(ActionType.SKILL_TAUNT, cooldown)

        elif action == ActionType.SKILL_PIERCE:
            # 레인저 전용: 관통샷
            if unit.role == RoleType.RANGER:
                if unit.can_use_skill(ActionType.SKILL_PIERCE):
                    cost = getattr(self.config, 'skill_pierce_cost', 25)
                    if unit.use_mp(cost):
                        self._execute_pierce(agent_id)
                        unit.set_cooldown(ActionType.SKILL_PIERCE, cooldown)

        elif action == ActionType.SKILL_BUFF:
            # 서포터 전용: 버프
            if unit.role == RoleType.SUPPORT:
                if unit.can_use_skill(ActionType.SKILL_BUFF):
                    cost = getattr(self.config, 'skill_buff_cost', 20)
                    if unit.use_mp(cost):
                        self._execute_buff(agent_id)
                        unit.set_cooldown(ActionType.SKILL_BUFF, cooldown)

    def _get_enemy_units(self, team_id: int) -> List[Unit]:
        """적 팀 유닛 목록 반환"""
        enemies = []
        enemy_list = self.team_b if team_id == 0 else self.team_a
        for agent_id in enemy_list:
            unit = self.units[agent_id]
            if unit.is_alive:
                enemies.append(unit)
        return enemies

    def _get_ally_units(self, team_id: int, exclude_id: Optional[str] = None) -> List[Unit]:
        """아군 유닛 목록 반환"""
        allies = []
        ally_list = self.team_a if team_id == 0 else self.team_b
        for agent_id in ally_list:
            if agent_id != exclude_id:
                unit = self.units[agent_id]
                if unit.is_alive:
                    allies.append(unit)
        return allies

    def _find_nearest_enemy(self, unit: Unit, enemies: List[Unit]) -> Optional[Unit]:
        """가장 가까운 적 찾기"""
        if not enemies:
            return None

        nearest = min(enemies, key=lambda e: unit.distance_to(e))
        return nearest

    def _find_lowest_hp_enemy(self, unit: Unit, enemies: List[Unit]) -> Optional[Unit]:
        """HP 가장 낮은 적 찾기"""
        if not enemies:
            return None

        # 공격 범위 내의 적만 고려
        in_range = [e for e in enemies if unit.distance_to(e) <= unit.attack_range]
        if not in_range:
            return None

        return min(in_range, key=lambda e: e.hp)

    def _find_attack_target(self, unit: Unit, enemies: List[Unit], prefer_nearest: bool = True) -> Optional[Unit]:
        """공격 타겟 찾기 (도발 상태면 도발한 탱커 우선)"""
        if not enemies:
            return None

        # 도발 상태인지 확인
        if unit.is_taunted():
            taunter_id = unit.taunted_by
            if taunter_id and taunter_id in self.units:
                taunter = self.units[taunter_id]
                if taunter.is_alive and unit.can_attack(taunter):
                    return taunter

        # 도발 상태가 아니거나 도발한 탱커를 공격할 수 없으면 기존 로직
        if prefer_nearest:
            return self._find_nearest_enemy(unit, enemies)
        else:
            return self._find_lowest_hp_enemy(unit, enemies)

    def _execute_attack(self, attacker_id: str, target: Unit):
        """기본 공격 실행"""
        attacker = self.units[attacker_id]
        damage = attacker.get_attack_damage()
        actual_damage = target.take_damage(damage)

        # 공격자 이벤트 기록
        self.step_events[attacker_id].append({
            'type': 'damage_dealt',
            'amount': actual_damage
        })

        # 피격자 이벤트 기록 (탱커 보상용)
        for agent_id, unit in self.units.items():
            if unit == target:
                self.step_events[agent_id].append({
                    'type': 'damage_taken',
                    'amount': actual_damage
                })
                break

        if not target.is_alive:
            self.step_events[attacker_id].append({
                'type': 'kill'
            })
            # 죽은 유닛의 에이전트 ID 찾기
            for agent_id, unit in self.units.items():
                if unit == target:
                    self.step_events[agent_id].append({
                        'type': 'death'
                    })
                    self.game_map.remove_unit(target.x, target.y)
                    break

    def _execute_aoe_attack(self, attacker_id: str):
        """범위 공격 실행"""
        attacker = self.units[attacker_id]
        enemies = self._get_enemy_units(attacker.team_id)

        total_damage = 0
        kills = 0

        for enemy in enemies:
            if attacker.distance_to(enemy) <= self.config.skill_aoe_range:
                actual_damage = enemy.take_damage(self.config.skill_aoe_damage)
                total_damage += actual_damage

                if not enemy.is_alive:
                    kills += 1
                    # 죽은 유닛 처리
                    for agent_id, unit in self.units.items():
                        if unit == enemy:
                            self.step_events[agent_id].append({'type': 'death'})
                            self.game_map.remove_unit(enemy.x, enemy.y)
                            break

        self.step_events[attacker_id].append({
            'type': 'damage_dealt',
            'amount': total_damage
        })
        for _ in range(kills):
            self.step_events[attacker_id].append({'type': 'kill'})

    def _execute_aoe_heal(self, healer_id: str):
        """힐러 전용: 범위 힐 실행"""
        healer = self.units[healer_id]
        allies = self._get_ally_units(healer.team_id, exclude_id=healer_id)

        heal_amount = self.config.skill_heal_amount
        heal_range = self.config.skill_heal_range
        total_healed = 0
        allies_healed = 0

        # 범위 내 모든 아군 힐
        for ally in allies:
            if healer.distance_to(ally) <= heal_range:
                # HP가 이미 최대면 스킵
                if ally.hp < ally.max_hp:
                    actual_heal = ally.heal(heal_amount)
                    total_healed += actual_heal
                    if actual_heal > 0:
                        allies_healed += 1

        # 자신도 힐
        self_heal = healer.heal(heal_amount // 2)
        total_healed += self_heal

        self.step_events[healer_id].append({
            'type': 'aoe_heal',
            'amount': total_healed,
            'allies_healed': allies_healed
        })

    def _execute_taunt(self, taunter_id: str):
        """탱커 전용: 도발 실행"""
        taunter = self.units[taunter_id]
        enemies = self._get_enemy_units(taunter.team_id)

        taunt_range = getattr(self.config, 'skill_taunt_range', 3)
        taunt_duration = getattr(self.config, 'skill_taunt_duration', 3)
        taunted_count = 0

        # 범위 내 모든 적에게 도발 적용
        for enemy in enemies:
            if taunter.distance_to(enemy) <= taunt_range:
                enemy.apply_taunt(taunter_id, taunt_duration)
                taunted_count += 1

        self.step_events[taunter_id].append({
            'type': 'taunt',
            'taunted_count': taunted_count
        })

    def _execute_pierce(self, ranger_id: str):
        """레인저 전용: 관통샷 실행 (타겟-레인저 일직선 방향)"""
        ranger = self.units[ranger_id]
        enemies = self._get_enemy_units(ranger.team_id)

        if not enemies:
            return

        # 가장 가까운 적을 타겟으로
        target = self._find_nearest_enemy(ranger, enemies)
        if not target:
            return

        pierce_damage = getattr(self.config, 'skill_pierce_damage', 18)
        pierce_range = getattr(self.config, 'skill_pierce_range', 5)

        # 방향 계산 (타겟 - 레인저)
        dx = target.x - ranger.x
        dy = target.y - ranger.y

        # 정규화 (일직선 방향)
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)

        total_damage = 0
        enemies_hit = 0
        kills = 0

        # 일직선 경로 상의 적들에게 데미지
        for dist in range(1, pierce_range + 1):
            check_x = ranger.x + dx * dist
            check_y = ranger.y + dy * dist

            # 맵 범위 체크
            if not (0 <= check_x < self.config.map_width and 0 <= check_y < self.config.map_height):
                break

            # 벽에 막히면 중단
            if self.game_map.get_tile(check_x, check_y) == TileType.WALL:
                break

            # 해당 위치의 적 찾기
            for enemy in enemies:
                if enemy.x == check_x and enemy.y == check_y and enemy.is_alive:
                    actual_damage = enemy.take_damage(pierce_damage)
                    total_damage += actual_damage
                    enemies_hit += 1

                    if not enemy.is_alive:
                        kills += 1
                        for agent_id, unit in self.units.items():
                            if unit == enemy:
                                self.step_events[agent_id].append({'type': 'death'})
                                self.game_map.remove_unit(enemy.x, enemy.y)
                                break

        self.step_events[ranger_id].append({
            'type': 'pierce',
            'damage_dealt': total_damage,
            'enemies_hit': enemies_hit
        })
        for _ in range(kills):
            self.step_events[ranger_id].append({'type': 'kill'})

    def _execute_buff(self, supporter_id: str):
        """서포터 전용: 버프 실행"""
        supporter = self.units[supporter_id]
        allies = self._get_ally_units(supporter.team_id, exclude_id=supporter_id)

        buff_range = getattr(self.config, 'skill_buff_range', 3)
        buff_value = getattr(self.config, 'skill_buff_value', 10)
        buff_duration = getattr(self.config, 'skill_buff_duration', 5)
        allies_buffed = 0

        # 범위 내 아군에게 공격력 버프 (버프 없는 아군 우선)
        for ally in allies:
            if supporter.distance_to(ally) <= buff_range:
                # 이미 버프가 있으면 스킵 (중복 방지)
                if 'attack' not in ally.buffs:
                    ally.apply_buff_from(supporter_id, 'attack', buff_value, buff_duration)
                    allies_buffed += 1

        self.step_events[supporter_id].append({
            'type': 'buff',
            'allies_buffed': allies_buffed
        })

    def _apply_terrain_effects(self):
        """지형 효과 적용"""
        for agent_id, unit in self.units.items():
            if not unit.is_alive:
                continue

            tile = self.game_map.get_tile(unit.x, unit.y)

            if tile == TileType.DANGER:
                # 위험 타일: 데미지
                unit.take_damage(5)
                self.step_events[agent_id].append({
                    'type': 'danger_tile'
                })

            elif tile == TileType.BUFF_ATK:
                # 공격력 버프
                unit.apply_buff('attack', 5, 3)

            elif tile == TileType.BUFF_DEF:
                # 방어력 버프
                unit.apply_buff('defense', 5, 3)

            elif tile == TileType.BUFF_HEAL:
                # 힐 버프
                unit.heal(10)

    def _update_unit_states(self):
        """유닛 상태 업데이트"""
        for unit in self.units.values():
            if unit.is_alive:
                unit.update_cooldowns()
                unit.update_buffs()
                unit.update_taunt()  # 도발 상태 업데이트
                # MP 자연 회복
                unit.recover_mp(5)

    def _check_game_end(self):
        """게임 종료 조건 체크"""
        team_a_alive = sum(1 for aid in self.team_a if self.units[aid].is_alive)
        team_b_alive = sum(1 for aid in self.team_b if self.units[aid].is_alive)

        if team_a_alive == 0:
            self.done = True
            self.winner = 1  # Team B 승리
        elif team_b_alive == 0:
            self.done = True
            self.winner = 0  # Team A 승리
        elif self.current_step >= self.config.max_steps:
            self.done = True
            # HP 총합으로 승자 결정
            team_a_hp = sum(self.units[aid].hp for aid in self.team_a)
            team_b_hp = sum(self.units[aid].hp for aid in self.team_b)
            if team_a_hp > team_b_hp:
                self.winner = 0
            elif team_b_hp > team_a_hp:
                self.winner = 1
            else:
                self.winner = None  # 무승부

    def _calculate_rewards(self) -> Dict[str, float]:
        """보상 계산 (역할별 차별화 포함)"""
        rewards = {}

        for agent_id, unit in self.units.items():
            reward = 0.0
            damage_dealt = 0
            heal_done = 0
            damage_taken = 0

            # 개인 보상 (이벤트 기반)
            for event in self.step_events.get(agent_id, []):
                if event['type'] == 'damage_dealt':
                    damage_dealt += event['amount']
                    reward += self.config.reward_damage * event['amount']
                elif event['type'] == 'kill':
                    reward += self.config.reward_kill
                elif event['type'] == 'death':
                    reward += self.config.reward_death
                    # 역할별 사망 추가 패널티
                    if unit.role == RoleType.TANK:
                        # 탱커: 흡수 데미지에 따라 패널티 감소
                        base_penalty = getattr(self.config, 'reward_tank_death', -2.0)
                        absorb_bonus = getattr(self.config, 'reward_tank_absorb_bonus', 0.02)
                        damage_reduction = min(abs(base_penalty) * 0.5, unit.total_damage_taken * absorb_bonus)
                        reward += base_penalty + damage_reduction
                    elif unit.role == RoleType.DEALER:
                        # 딜러: 화력 손실
                        reward += getattr(self.config, 'reward_dealer_death', -1.0)
                    elif unit.role == RoleType.HEALER:
                        # 힐러: 지속력 손실 (가장 중요)
                        reward += getattr(self.config, 'reward_healer_death', -1.5)
                elif event['type'] == 'heal':
                    heal_done += event['amount']
                    reward += self.config.reward_heal * event['amount']
                elif event['type'] == 'aoe_heal':
                    # 힐러: 범위힐 - 힐한 아군 수만큼 보상
                    heal_done += event['amount']
                    reward += self.config.reward_heal * event['amount']
                    allies_healed = event.get('allies_healed', 0)
                    reward += getattr(self.config, 'reward_heal_per_ally', 0.8) * allies_healed
                elif event['type'] == 'taunt':
                    # 탱커: 도발 - 도발된 적 수만큼 보상
                    taunted_count = event.get('taunted_count', 0)
                    reward += getattr(self.config, 'reward_taunt_per_enemy', 1.0) * taunted_count
                elif event['type'] == 'pierce':
                    # 레인저: 관통샷 - 적중 적 수만큼 보상
                    damage_dealt += event.get('damage_dealt', 0)
                    reward += self.config.reward_damage * event.get('damage_dealt', 0)
                    enemies_hit = event.get('enemies_hit', 0)
                    reward += getattr(self.config, 'reward_pierce_per_enemy', 0.7) * enemies_hit
                elif event['type'] == 'buff':
                    # 서포터: 버프 - 버프한 아군 수만큼 보상
                    allies_buffed = event.get('allies_buffed', 0)
                    reward += getattr(self.config, 'reward_buff_ongoing', 0.3) * allies_buffed * 2
                elif event['type'] == 'danger_tile':
                    reward += self.config.reward_danger_tile
                elif event['type'] == 'damage_taken':
                    damage_taken += event['amount']

            # 제자리 패널티 / 접근 보상
            for event in self.step_events.get(agent_id, []):
                if event['type'] == 'stay':
                    reward += self.config.reward_stay
                elif event['type'] == 'approach':
                    reward += self.config.reward_approach

            # 매 턴 시간 패널티
            reward += self.config.reward_time_penalty

            # === 역할별 추가 보상 ===
            if unit.is_alive:
                enemies = self._get_enemy_units(unit.team_id)
                allies = self._get_ally_units(unit.team_id, exclude_id=agent_id)

                # 전투 범위 내 보상
                in_combat = any(unit.distance_to(e) <= unit.attack_range for e in enemies)
                if in_combat:
                    reward += self.config.reward_in_combat

                # === 존버 방지: 적과 너무 멀면 패널티 (양팀 공통) ===
                if enemies:
                    min_enemy_dist = min(unit.distance_to(e) for e in enemies)
                    # 적과 거리 8 초과면 패널티 (맵 크기 20 기준)
                    if min_enemy_dist > 8:
                        reward -= 0.1 * (min_enemy_dist - 8)  # 거리당 패널티

                # 탱커: 데미지 받으면 보상
                if unit.role == RoleType.TANK and damage_taken > 0:
                    reward += self.config.reward_tank_aggro * damage_taken

                # 딜러: 데미지 주면 추가 보상
                if unit.role == RoleType.DEALER and damage_dealt > 0:
                    reward += self.config.reward_dealer_damage * damage_dealt

                # 힐러: 힐하면 추가 보상
                if unit.role == RoleType.HEALER and heal_done > 0:
                    reward += self.config.reward_healer_heal * heal_done

                # 레인저: 원거리 공격 시 보상
                if unit.role == RoleType.RANGER and damage_dealt > 0:
                    if enemies:
                        min_dist = min(unit.distance_to(e) for e in enemies)
                        if min_dist >= 2:
                            reward += self.config.reward_ranger_range_atk * damage_dealt

                # 서포터: 아군 근처에 있으면 보상
                if unit.role == RoleType.SUPPORT:
                    nearby_allies = sum(1 for a in allies if unit.distance_to(a) <= 3)
                    reward += self.config.reward_support_nearby * nearby_allies

                    # 서포터 버프 지속 보상: 버프받은 아군이 살아있으면 매 턴 보상
                    buff_ongoing_reward = getattr(self.config, 'reward_buff_ongoing', 0.3)
                    for ally in allies:
                        if ally.buffed_by == agent_id and ally.is_alive:
                            reward += buff_ongoing_reward

                # === v7 단순화 모드: 탱커 근처에서 붙어서 공격 ===
                use_v7 = getattr(self.config, 'use_v7_simple', False)
                if use_v7:
                    # 보상 설정 (붙어서 공격!)
                    reward_per_tile = getattr(self.config, 'reward_per_tile', 1.5)  # 칸당 보상
                    reward_leave_per_tile = getattr(self.config, 'reward_leave_per_tile', -2.0)  # 이탈 패널티 (완화)
                    reward_combat = getattr(self.config, 'reward_combat', 5.0)  # 탱커 근처 공격 보상 (높음!)
                    reward_behind_tank = getattr(self.config, 'reward_behind_tank', 0.5)  # 탱커 뒤 위치 보상
                    proximity_threshold = getattr(self.config, 'proximity_threshold', 2)  # 2칸 이내 = 3x3

                    # A팀만 협동 보상 (B팀은 고정 모델)
                    if unit.team_id == 0 and self.config.player_idx >= 0:
                        tank_id = f"team_a_{self.config.player_idx}"
                        if tank_id in self.units and agent_id != tank_id:
                            tank = self.units[tank_id]
                            if tank.is_alive:
                                dist = unit.distance_to(tank)

                                # 1. 근접 유지 보상 (2칸 이내: 거리 반비례)
                                if dist <= proximity_threshold:
                                    # 1칸: +3, 2칸: +1.5
                                    reward += reward_per_tile * (proximity_threshold + 1 - dist)
                                # 2. 이탈 패널티 (3칸+: 거리당 패널티)
                                else:
                                    # 3칸: -2, 4칸: -4, 5칸: -6 ... (완화됨)
                                    excess_dist = dist - proximity_threshold
                                    reward += reward_leave_per_tile * excess_dist

                                # 3. 공격 보상 (탱커 근처에서만!)
                                combat_range = getattr(self.config, 'combat_range', 3)
                                if damage_dealt > 0 and dist <= combat_range:
                                    # 탱커 3칸 이내에서 공격해야 보상
                                    reward += reward_combat
                                elif damage_dealt > 0 and dist > combat_range:
                                    # 탱커에서 멀리서 공격하면 패널티!
                                    reward -= 1.0

                                # 4. 탱커 뒤 위치 보상 (탱커보다 적과 멀면)
                                if enemies:
                                    nearest_enemy = min(enemies, key=lambda e: tank.distance_to(e))
                                    tank_to_enemy = tank.distance_to(nearest_enemy)
                                    unit_to_enemy = unit.distance_to(nearest_enemy)
                                    # NPC가 탱커보다 적과 멀면 = 탱커 뒤에 있음
                                    if unit_to_enemy > tank_to_enemy:
                                        reward += reward_behind_tank

                                # 5. 시간 패널티 (존버 방지)
                                time_penalty = getattr(self.config, 'reward_time_penalty', -0.05)
                                reward += time_penalty

                    # v7 게임 종료 보상
                    if self.done:
                        if self.winner == unit.team_id:
                            reward += self.config.reward_win
                        elif self.winner is not None:
                            reward += self.config.reward_lose

                    # v7 모드일 때 기존 협동 로직 건너뛰기
                    rewards[agent_id] = reward
                    continue

                # === v6 협동 보상 (use_v6_coop 플래그 확인) ===
                use_v6 = getattr(self.config, 'use_v6_coop', False)

                # === 플레이어 협동 보상 (A팀, player_idx >= 0일 때만) ===
                if self.config.player_idx >= 0 and unit.team_id == 0:
                    player_agent_id = f"team_a_{self.config.player_idx}"
                    if player_agent_id in self.units and agent_id != player_agent_id:
                        player_unit = self.units[player_agent_id]
                        if player_unit.is_alive:
                            dist_to_player = unit.distance_to(player_unit)
                            player_role = player_unit.role

                            # === v6: 역할별 협동 거리 ===
                            if use_v6:
                                role_thresholds = {
                                    RoleType.TANK: getattr(self.config, 'coop_threshold_tank', 3),
                                    RoleType.DEALER: getattr(self.config, 'coop_threshold_dealer', 2),
                                    RoleType.HEALER: getattr(self.config, 'coop_threshold_healer', 4),
                                    RoleType.RANGER: getattr(self.config, 'coop_threshold_ranger', 6),
                                    RoleType.SUPPORT: getattr(self.config, 'coop_threshold_support', 4),
                                }
                                coop_threshold = role_thresholds.get(unit.role, 4)
                            else:
                                coop_threshold = getattr(self.config, 'coop_distance_threshold', 5)

                            # === 조건부 협동: 전투 중이거나 행동했을 때만 보상 ===
                            is_fighting = in_combat or damage_dealt > 0 or heal_done > 0

                            # === v6: 3단계 명확한 보상 ===
                            if use_v6:
                                precombat_reward = getattr(self.config, 'reward_near_player_precombat', 1.0)
                                combat_reward = getattr(self.config, 'reward_near_player_combat', 3.0)
                                wander_penalty = getattr(self.config, 'reward_wander_penalty', -1.5)

                                if is_fighting:
                                    # === 1단계: 전투 중 → 강한 + 보상 ===
                                    if dist_to_player <= coop_threshold:
                                        reward += combat_reward  # 전투 + 근처 = 최고
                                    else:
                                        reward += combat_reward * 0.5  # 전투만 해도 보상
                                elif dist_to_player <= coop_threshold:
                                    # === 2단계: 전투 없지만 플레이어 근처 → 약한 + 보상 ===
                                    reward += precombat_reward
                                else:
                                    # === 3단계: 전투 없고 플레이어 멀리 → 패널티! ===
                                    reward += wander_penalty

                                # === v6: 밀집 패널티 (AoE 방지) ===
                                min_safe_dist = getattr(self.config, 'min_safe_distance', 1)
                                too_close_penalty = getattr(self.config, 'reward_too_close_penalty', -0.5)
                                if dist_to_player < min_safe_dist and unit.role != RoleType.TANK:
                                    reward += too_close_penalty

                                # === v6: 탱커 방향 보상/패널티 (가장 가까운 적 기준) ===
                                if unit.role == RoleType.TANK and enemies:
                                    tank_front_bonus = getattr(self.config, 'reward_tank_front_bonus', 2.0)
                                    tank_behind_penalty = getattr(self.config, 'reward_tank_behind_penalty', -1.0)

                                    # 가장 가까운 적 기준
                                    nearest_enemy = min(enemies, key=lambda e: unit.distance_to(e))
                                    player_to_enemy = player_unit.distance_to(nearest_enemy)
                                    tank_to_enemy = unit.distance_to(nearest_enemy)

                                    if tank_to_enemy < player_to_enemy and dist_to_player <= 3:
                                        # 탱커가 플레이어 앞에 있음 → 보상!
                                        reward += tank_front_bonus
                                    elif tank_to_enemy > player_to_enemy:
                                        # 탱커가 플레이어 뒤에 있음 → 패널티!
                                        reward += tank_behind_penalty

                                    # === v6: 탱커가 딜러/힐러보다 뒤에 있으면 패널티 ===
                                    for ally in allies:
                                        if ally.role in [RoleType.DEALER, RoleType.HEALER] and ally.is_alive:
                                            ally_to_enemy = ally.distance_to(nearest_enemy)
                                            if tank_to_enemy > ally_to_enemy:
                                                # 탱커가 딜러/힐러보다 뒤에 있음 → 패널티!
                                                reward += tank_behind_penalty

                            else:
                                # === 기존 v2 로직 ===
                                # 1. 거리 기반 연속 보상 (전투 중일 때만!)
                                if dist_to_player <= coop_threshold and is_fighting:
                                    proximity_reward = getattr(self.config, 'reward_approach_player', 0.05)
                                    distance_bonus = proximity_reward * (coop_threshold - dist_to_player) / coop_threshold
                                    reward += distance_bonus

                                # 2. 플레이어 근처 보상 (거리 3 이내 + 전투 중)
                                if dist_to_player <= 2 and is_fighting:
                                    reward += self.config.reward_near_player

                            # 3. 탱커가 플레이어(딜러/힐러/레인저) 보호
                            if unit.role == RoleType.TANK:
                                if player_role in [RoleType.DEALER, RoleType.HEALER, RoleType.RANGER]:
                                    # 탱커가 플레이어와 적 사이에 있으면 보상
                                    for enemy in enemies:
                                        player_to_enemy = player_unit.distance_to(enemy)
                                        tank_to_enemy = unit.distance_to(enemy)
                                        if tank_to_enemy < player_to_enemy and dist_to_player <= 3:
                                            reward += self.config.reward_protect_player
                                            break

                            # 4. 힐러가 플레이어 근처에서 힐하면 추가 보상
                            if unit.role == RoleType.HEALER and heal_done > 0:
                                if dist_to_player <= self.config.skill_heal_range:
                                    reward += self.config.reward_support_player

                            # 5. 힐러: 플레이어 HP 낮으면 다가가도록 유도
                            if unit.role == RoleType.HEALER:
                                if player_unit.hp / player_unit.max_hp < 0.5:
                                    if dist_to_player <= self.config.skill_heal_range:
                                        reward += self.config.reward_support_player * 0.5

                            # 6. 딜러/힐러/레인저가 탱커 뒤에 있으면 보상
                            if unit.role in [RoleType.DEALER, RoleType.HEALER, RoleType.RANGER]:
                                tank_unit = None
                                for ally in allies:
                                    if ally.role == RoleType.TANK:
                                        tank_unit = ally
                                        break
                                if tank_unit and tank_unit.is_alive:
                                    for enemy in enemies:
                                        tank_to_enemy = tank_unit.distance_to(enemy)
                                        unit_to_enemy = unit.distance_to(enemy)
                                        if tank_to_enemy < unit_to_enemy:
                                            reward += self.config.reward_follow_tank
                                            break

                            # 7. 플레이어가 탱커일 때: 다른 유닛이 플레이어 뒤에 있으면 보상
                            if player_role == RoleType.TANK:
                                if unit.role in [RoleType.DEALER, RoleType.HEALER, RoleType.RANGER]:
                                    for enemy in enemies:
                                        player_to_enemy = player_unit.distance_to(enemy)
                                        unit_to_enemy = unit.distance_to(enemy)
                                        if player_to_enemy < unit_to_enemy and dist_to_player <= 4:
                                            reward += self.config.reward_follow_tank
                                            break

                # === B팀 플레이어 협동 보상 (player_idx_b >= 0일 때) ===
                # === v2 수정: 조건부 협동 보상 - 전투 중일 때만 ===
                player_idx_b = getattr(self.config, 'player_idx_b', -1)
                if player_idx_b >= 0 and unit.team_id == 1:
                    player_agent_id_b = f"team_b_{player_idx_b}"
                    if player_agent_id_b in self.units and agent_id != player_agent_id_b:
                        player_unit_b = self.units[player_agent_id_b]
                        if player_unit_b.is_alive:
                            dist_to_player = unit.distance_to(player_unit_b)
                            player_role = player_unit_b.role
                            coop_threshold = getattr(self.config, 'coop_distance_threshold', 5)

                            # === 조건부 협동: 전투 중이거나 행동했을 때만 보상 ===
                            is_fighting = in_combat or damage_dealt > 0 or heal_done > 0

                            # 1. 거리 기반 연속 보상 (전투 중일 때만!)
                            if dist_to_player <= coop_threshold and is_fighting:
                                proximity_reward = getattr(self.config, 'reward_approach_player', 0.05)
                                distance_bonus = proximity_reward * (coop_threshold - dist_to_player) / coop_threshold
                                reward += distance_bonus

                            # 2. 플레이어 근처 보상 (전투 중일 때만!)
                            if dist_to_player <= 2 and is_fighting:
                                reward += self.config.reward_near_player

                            # 3. 탱커가 플레이어 보호
                            if unit.role == RoleType.TANK:
                                if player_role in [RoleType.DEALER, RoleType.HEALER, RoleType.RANGER]:
                                    for enemy in enemies:
                                        player_to_enemy = player_unit_b.distance_to(enemy)
                                        tank_to_enemy = unit.distance_to(enemy)
                                        if tank_to_enemy < player_to_enemy and dist_to_player <= 2:
                                            reward += self.config.reward_protect_player
                                            break

                            # 4. 힐러가 플레이어 근처에서 힐
                            if unit.role == RoleType.HEALER and heal_done > 0:
                                if dist_to_player <= self.config.skill_heal_range:
                                    reward += self.config.reward_support_player

                            # 5. 힐러: 플레이어 HP 낮으면 다가가기
                            if unit.role == RoleType.HEALER:
                                if player_unit_b.hp / player_unit_b.max_hp < 0.5:
                                    if dist_to_player <= self.config.skill_heal_range:
                                        reward += self.config.reward_support_player * 0.5

                            # 6. 딜러/힐러/레인저가 탱커 뒤에 있으면 보상
                            if unit.role in [RoleType.DEALER, RoleType.HEALER, RoleType.RANGER]:
                                tank_unit = None
                                for ally in allies:
                                    if ally.role == RoleType.TANK:
                                        tank_unit = ally
                                        break
                                if tank_unit and tank_unit.is_alive:
                                    for enemy in enemies:
                                        tank_to_enemy = tank_unit.distance_to(enemy)
                                        unit_to_enemy = unit.distance_to(enemy)
                                        if tank_to_enemy < unit_to_enemy:
                                            reward += self.config.reward_follow_tank
                                            break

                            # 7. 플레이어가 탱커일 때
                            if player_role == RoleType.TANK:
                                if unit.role in [RoleType.DEALER, RoleType.HEALER, RoleType.RANGER]:
                                    for enemy in enemies:
                                        player_to_enemy = player_unit_b.distance_to(enemy)
                                        unit_to_enemy = unit.distance_to(enemy)
                                        if player_to_enemy < unit_to_enemy and dist_to_player <= 4:
                                            reward += self.config.reward_follow_tank
                                            break

                # === B팀 기본 협동 보상 (player_idx_b < 0일 때 - AI끼리 협동) ===
                elif unit.team_id == 1:
                    # B팀 내 딜러/힐러/레인저 찾기 (보호 대상)
                    protect_targets = [a for a in allies if a.role in [RoleType.DEALER, RoleType.HEALER, RoleType.RANGER]]

                    # 1. 탱커: 딜러/힐러/레인저 보호
                    if unit.role == RoleType.TANK:
                        for target in protect_targets:
                            for enemy in enemies:
                                target_to_enemy = target.distance_to(enemy)
                                tank_to_enemy = unit.distance_to(enemy)
                                if tank_to_enemy < target_to_enemy and tank_to_enemy <= 5:
                                    reward += self.config.reward_protect_player
                                    break

                    # 2. 힐러: 팀원 힐하면 추가 보상
                    if unit.role == RoleType.HEALER and heal_done > 0:
                        if enemies:
                            min_enemy_dist = min(unit.distance_to(e) for e in enemies)
                            if min_enemy_dist <= 6:
                                reward += self.config.reward_support_player

                    # 3. 딜러/레인저: 탱커 뒤에 있기
                    if unit.role in [RoleType.DEALER, RoleType.RANGER]:
                        tank_unit = None
                        for ally in allies:
                            if ally.role == RoleType.TANK:
                                tank_unit = ally
                                break
                        if tank_unit and tank_unit.is_alive and enemies:
                            min_enemy_dist = min(unit.distance_to(e) for e in enemies)
                            if min_enemy_dist <= 6:
                                for enemy in enemies:
                                    tank_to_enemy = tank_unit.distance_to(enemy)
                                    unit_to_enemy = unit.distance_to(enemy)
                                    if tank_to_enemy < unit_to_enemy:
                                        reward += self.config.reward_follow_tank
                                        break

                # === 존버 방지: 적과 너무 멀면 패널티 (양팀 모두) ===
                if enemies:
                    min_enemy_dist = min(unit.distance_to(e) for e in enemies)
                    if min_enemy_dist > 8:
                        reward -= 0.1 * (min_enemy_dist - 8)

            # 게임 종료 보상
            if self.done:
                if self.winner == unit.team_id:
                    reward += self.config.reward_win
                elif self.winner is not None:
                    reward += self.config.reward_lose
                else:
                    reward += self.config.reward_draw

            rewards[agent_id] = reward

        return rewards

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """모든 에이전트의 관찰 생성"""
        observations = {}
        for agent_id in self.units:
            observations[agent_id] = self._get_single_observation(agent_id)
        return observations

    def _get_single_observation(self, agent_id: str) -> np.ndarray:
        """단일 에이전트의 관찰 벡터 생성"""
        unit = self.units[agent_id]
        obs_parts = []

        # 1. 자기 상태
        self_state = unit.get_state_vector()
        obs_parts.append(self_state)

        # 2. 아군 상태 (4명, 패딩)
        allies = self._get_ally_units(unit.team_id, exclude_id=agent_id)
        ally_states = []
        for i in range(4):
            if i < len(allies):
                ally_states.append(allies[i].get_state_vector())
            else:
                # 패딩 (죽었거나 없는 아군)
                ally_states.append(np.zeros(10))
        obs_parts.extend(ally_states)

        # 3. 적군 상태 (5명)
        enemies = self._get_enemy_units(unit.team_id)
        enemy_states = []
        for i in range(5):
            if i < len(enemies):
                enemy_states.append(enemies[i].get_state_vector())
            else:
                enemy_states.append(np.zeros(10))
        obs_parts.extend(enemy_states)

        # 4. 지형 정보 (주변 패치, 평탄화)
        terrain = self.game_map.get_local_observation(
            unit.x, unit.y, self.config.obs_range
        )
        obs_parts.append(terrain.flatten() / 5.0)  # 정규화

        # 5. 전역 정보
        global_info = np.array([
            min(self.current_step / self.config.max_steps, 1.0),  # 정규화된 턴 수 (clamp to 1.0)
            unit.team_id  # 팀 ID
        ])
        obs_parts.append(global_info)

        # 6. 플레이어(탱커) 정보 - 협동 학습용
        player_info = np.zeros(6)
        player_idx = getattr(self.config, 'player_idx', -1)

        if player_idx >= 0:
            if unit.team_id == 0:
                tank_id = f"team_a_{player_idx}"
                if tank_id in self.units and agent_id != tank_id:
                    tank = self.units[tank_id]
                    player_info = np.array([
                        (tank.x - unit.x) / self.config.map_width,
                        (tank.y - unit.y) / self.config.map_height,
                        unit.distance_to(tank) / 20.0,
                        float(tank.is_alive),
                        tank.hp / tank.max_hp if tank.max_hp > 0 else 0,
                        1.0
                    ])

        obs_parts.append(player_info)

        # 연결 (obs_size=223, v7 coop용)
        return np.concatenate(obs_parts).astype(np.float32)

    def _get_zero_rewards(self) -> Dict[str, float]:
        return {agent_id: 0.0 for agent_id in self.units}

    def _get_dones(self) -> Dict[str, bool]:
        return {agent_id: self.done for agent_id in self.units}

    def _get_truncated(self) -> Dict[str, bool]:
        truncated = self.current_step >= self.config.max_steps and not self.done
        return {agent_id: truncated for agent_id in self.units}

    def _get_infos(self) -> Dict[str, Any]:
        return {
            'step': self.current_step,
            'winner': self.winner,
            'team_a_alive': sum(1 for aid in self.team_a if self.units[aid].is_alive),
            'team_b_alive': sum(1 for aid in self.team_b if self.units[aid].is_alive),
        }

    def get_agent_ids(self) -> List[str]:
        """모든 에이전트 ID 반환"""
        return list(self.units.keys())

    def render(self) -> np.ndarray:
        """렌더링용 상태 반환 (시각화에서 사용)"""
        state = {
            'map': self.game_map.tiles.copy(),
            'units': {},
            'step': self.current_step,
            'winner': self.winner,
        }

        for agent_id, unit in self.units.items():
            state['units'][agent_id] = {
                'x': unit.x,
                'y': unit.y,
                'hp': unit.hp,
                'max_hp': unit.max_hp,
                'team_id': unit.team_id,
                'role': unit.role,
                'is_alive': unit.is_alive,
            }

        return state
