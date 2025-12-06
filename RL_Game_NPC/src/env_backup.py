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

        self.obs_size = self_state_size + ally_state_size + enemy_state_size + terrain_size + global_size

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

        # Team A 유닛 생성
        spawn_a = self.game_map.get_spawn_positions(0, self.config.team_size)
        for i, (x, y) in enumerate(spawn_a):
            agent_id = f"team_a_{i}"
            role = self.config.team_composition[i]
            unit = Unit(unit_id=i, team_id=0, role=role, x=x, y=y)
            self.units[agent_id] = unit
            self.team_a.append(agent_id)
            self.game_map.place_unit(agent_id, x, y)

        # Team B 유닛 생성
        spawn_b = self.game_map.get_spawn_positions(1, self.config.team_size)
        for i, (x, y) in enumerate(spawn_b):
            agent_id = f"team_b_{i}"
            role = self.config.team_composition[i]
            unit = Unit(unit_id=i, team_id=1, role=role, x=x, y=y)
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
        """공격/스킬 행동 적용"""
        unit = self.units[agent_id]
        enemies = self._get_enemy_units(unit.team_id)

        if action == ActionType.ATTACK_NEAREST:
            # 가장 가까운 적 공격
            target = self._find_nearest_enemy(unit, enemies)
            if target and unit.can_attack(target):
                self._execute_attack(agent_id, target)

        elif action == ActionType.ATTACK_LOWEST:
            # HP 가장 낮은 적 공격
            target = self._find_lowest_hp_enemy(unit, enemies)
            if target and unit.can_attack(target):
                self._execute_attack(agent_id, target)

        elif action == ActionType.SKILL_AOE:
            # 범위 공격
            if unit.can_use_skill(ActionType.SKILL_AOE):
                if unit.use_mp(self.config.skill_aoe_cost):
                    self._execute_aoe_attack(agent_id)
                    unit.set_cooldown(ActionType.SKILL_AOE, 3)

        elif action == ActionType.SKILL_HEAL:
            # 힐 스킬 (힐러만 효과적)
            if unit.can_use_skill(ActionType.SKILL_HEAL):
                if unit.use_mp(self.config.skill_heal_cost):
                    self._execute_heal(agent_id)
                    unit.set_cooldown(ActionType.SKILL_HEAL, 2)

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

    def _execute_attack(self, attacker_id: str, target: Unit):
        """기본 공격 실행"""
        attacker = self.units[attacker_id]
        damage = attacker.get_attack_damage()
        actual_damage = target.take_damage(damage)

        # 이벤트 기록
        self.step_events[attacker_id].append({
            'type': 'damage_dealt',
            'amount': actual_damage
        })

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

    def _execute_heal(self, healer_id: str):
        """힐 스킬 실행"""
        healer = self.units[healer_id]
        allies = self._get_ally_units(healer.team_id, exclude_id=healer_id)

        # 힐러 역할이면 힐량 보너스
        heal_amount = self.config.skill_heal_amount
        if healer.role == RoleType.HEALER:
            heal_amount = int(heal_amount * 1.5)

        total_healed = 0

        # HP가 가장 낮은 아군에게 힐
        if allies:
            target = min(allies, key=lambda a: a.hp / a.max_hp)
            if healer.distance_to(target) <= self.config.skill_heal_range:
                actual_heal = target.heal(heal_amount)
                total_healed = actual_heal

        # 자신도 소량 힐
        self_heal = healer.heal(heal_amount // 3)
        total_healed += self_heal

        self.step_events[healer_id].append({
            'type': 'heal',
            'amount': total_healed
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
        """보상 계산"""
        rewards = {}

        for agent_id, unit in self.units.items():
            reward = 0.0

            # 개인 보상 (이벤트 기반)
            for event in self.step_events.get(agent_id, []):
                if event['type'] == 'damage_dealt':
                    reward += self.config.reward_damage * event['amount']
                elif event['type'] == 'kill':
                    reward += self.config.reward_kill
                elif event['type'] == 'death':
                    reward += self.config.reward_death
                elif event['type'] == 'heal':
                    reward += self.config.reward_heal * event['amount']
                elif event['type'] == 'danger_tile':
                    reward += self.config.reward_danger_tile

            # 제자리 패널티
            for event in self.step_events.get(agent_id, []):
                if event['type'] == 'stay':
                    reward += self.config.reward_stay
                elif event['type'] == 'approach':
                    reward += self.config.reward_approach

            # 매 턴 시간 패널티
            reward += self.config.reward_time_penalty
            
            # 게임 종료 보상
            if self.done:
                if self.winner == unit.team_id:
                    reward += self.config.reward_win
                elif self.winner is not None:
                    reward += self.config.reward_lose
                else:
                    # 무승부 패널티
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
            self.current_step / self.config.max_steps,  # 정규화된 턴 수
            unit.team_id  # 팀 ID
        ])
        obs_parts.append(global_info)

        # 연결
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