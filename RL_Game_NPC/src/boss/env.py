"""BossRaidEnv — 1보스 4파티 레이드 환경

- 동시 행동 처리 (원 논문 방식 계승)
- 보스 FSM 1체 + 파티 유닛 4명
- 플레이어(딜러) 1명은 외부에서 action 주입받음
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import random

from .config import (
    BossConfig, PartyRole, PatternID, PhaseID, BossActionID,
    ROLE_STATS_BOSS,
)
from .boss import Boss
from .patterns import ActiveTelegraph, Pos
from .rewards import RewardComputer


# ─────────────────── 파티 유닛 ───────────────────

@dataclass
class PartyUnit:
    uid: int
    role: PartyRole
    x: int
    y: int
    hp: int
    mp: int
    max_hp: int
    max_mp: int
    attack: int
    defense: int
    attack_range: int
    alive: bool = True
    # 쿨타임 (스킬·도발·힐·버프 등)
    cooldowns: Dict[int, int] = field(default_factory=dict)
    # 버프 (남은 턴)
    buff_atk: int = 0
    buff_shield: int = 0
    buff_guard: int = 0
    # 디버프
    marked_turns: int = 0     # MARK 대상
    chained_with: Optional[int] = None
    chain_turns: int = 0
    # 통계
    total_damage_dealt: int = 0
    total_damage_taken: int = 0
    total_heal_done: int = 0


# ─────────────────── 환경 ───────────────────

class BossRaidEnv:
    """보스 레이드 환경.

    agent_ids: "p0", "p1", "p2", "p3" (파티 슬롯 순서 = DEALER, TANK, HEALER, SUPPORT)
    """

    def __init__(self, config: Optional[BossConfig] = None, seed: Optional[int] = None):
        self.config = config or BossConfig()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.units: Dict[int, PartyUnit] = {}
        self.boss: Optional[Boss] = None
        self.current_step = 0
        self.done = False
        self.wipe = False
        self.victory = False

        self.reward_computer = RewardComputer(self.config)
        self.step_events: Dict[int, List[dict]] = {}

        self.reset()

    # ────────────── 편의 ──────────────

    def agent_ids(self) -> List[str]:
        return [f"p{i}" for i in range(len(self.config.party_roles))]

    def uid_of(self, agent_id: str) -> int:
        return int(agent_id[1:])

    def party_positions(self) -> Dict[int, Pos]:
        return {u.uid: (u.x, u.y) for u in self.units.values() if u.alive}

    def party_roles(self) -> Dict[int, int]:
        return {u.uid: int(u.role) for u in self.units.values()}

    # ────────────── reset ──────────────

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.default_rng(seed)

        self.current_step = 0
        self.done = False
        self.wipe = False
        self.victory = False
        self.step_events.clear()

        # 파티 유닛 배치 — 보스 반대편 사분면에 모여 시작
        self.units.clear()
        w, h = self.config.map_width, self.config.map_height
        start_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for i, role in enumerate(self.config.party_roles):
            stats = ROLE_STATS_BOSS[role]
            sx, sy = start_positions[i]
            u = PartyUnit(
                uid=i, role=role, x=sx, y=sy,
                hp=stats.hp, mp=stats.mp,
                max_hp=stats.hp, max_mp=stats.mp,
                attack=stats.attack, defense=stats.defense,
                attack_range=stats.attack_range,
            )
            self.units[i] = u

        # 보스 초기화 (맵 중앙)
        self.boss = Boss(config=self.config, rng=self.rng)
        self.boss.hp = self.config.boss_max_hp

        return self._get_all_observations()

    # ────────────── step ──────────────

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        """actions: {"p0": action_id, ...}"""
        self.step_events = {uid: [] for uid in self.units}
        self.current_step += 1

        # 1. 보스: 페이즈 전이 체크
        phase_changed = self.boss.check_phase_transition()
        if phase_changed:
            for uid in self.units:
                self.step_events[uid].append({"type": "phase_clear"})

        # 2. 보스: 텔레그래프 카운트다운
        ready_telegraphs = self.boss.tick_telegraphs()

        # 3. 파티 행동 처리 (동시 처리)
        self._resolve_party_actions(actions)

        # 4. 보스 텔레그래프 impact 적용
        for tg in ready_telegraphs:
            self._apply_telegraph_impact(tg)
            tg.impacted = True

        # 5. 보스 기본 공격 (쿨 없음, 매 턴 가능, 단 텔레그래프 없거나 그로기 아닐 때)
        if self.boss.invuln_turns <= 0 and self.boss.grog_turns <= 0:
            # 신규 패턴 시전
            if len(self.boss.telegraphs) < self.config.max_active_telegraphs:
                pid = self.boss.select_pattern()
                if pid is not None:
                    self.boss.start_pattern(pid, self.party_positions(), self.party_roles())

        # 6. 지속형 디버프 (연결) 처리
        self._tick_chains()

        # 7. 보스 이동 — 어그로 1위 방향
        top_uid = self.boss.top_aggro_uid()
        if top_uid is not None and top_uid in self.units and self.units[top_uid].alive:
            self._move_boss_toward(self.units[top_uid].x, self.units[top_uid].y)

        # 8. 텔레그래프 정리
        self.boss.finalize_telegraphs()

        # 9. 버프/디버프 틱
        for u in self.units.values():
            if u.buff_atk > 0: u.buff_atk -= 1
            if u.buff_shield > 0: u.buff_shield -= 1
            if u.buff_guard > 0: u.buff_guard -= 1
            if u.marked_turns > 0: u.marked_turns -= 1
            for k in list(u.cooldowns.keys()):
                u.cooldowns[k] = max(0, u.cooldowns[k] - 1)

        # 10. 보스 end-of-turn
        self.boss.tick_end_of_turn()

        # 11. 종료 조건
        player_uid = self.config.player_slot
        self.victory = self.boss.hp <= 0
        self.wipe = all((not u.alive) for u in self.units.values())
        player_dead = not self.units[player_uid].alive
        self.done = (
            self.victory or self.wipe or player_dead or
            self.current_step >= self.config.max_steps
        )

        # 12. 관찰, 보상, done, info 반환
        obs = self._get_all_observations()
        rewards = self.reward_computer.compute(self)
        dones = {aid: self.done for aid in self.agent_ids()}
        infos = self._build_infos()
        return obs, rewards, dones, infos

    # ────────────── 파티 행동 처리 ──────────────

    def _resolve_party_actions(self, actions: Dict[str, int]):
        # 이동 먼저 (동시 이동 - 충돌 처리)
        move_targets: Dict[int, Pos] = {}
        for aid, action in actions.items():
            uid = self.uid_of(aid)
            u = self.units[uid]
            if not u.alive: continue
            move_targets[uid] = self._compute_move(u, action)

        # 충돌 처리: 같은 타일로 2명이 가면 먼저 선언한 쪽만 허용 (uid 낮은 순)
        occupied: Set[Pos] = set()
        boss_tiles = self._boss_tiles()
        occupied |= boss_tiles
        for uid in sorted(move_targets.keys()):
            tgt = move_targets[uid]
            if tgt in occupied:
                continue
            self.units[uid].x, self.units[uid].y = tgt
            occupied.add(tgt)

        # 비이동 액션
        for aid, action in actions.items():
            uid = self.uid_of(aid)
            u = self.units[uid]
            if not u.alive: continue
            self._execute_non_move(u, action)

    def _compute_move(self, u: PartyUnit, action: int) -> Pos:
        x, y = u.x, u.y
        if action == BossActionID.MOVE_UP:    y = max(0, y - 1)
        elif action == BossActionID.MOVE_DOWN: y = min(self.config.map_height - 1, y + 1)
        elif action == BossActionID.MOVE_LEFT: x = max(0, x - 1)
        elif action == BossActionID.MOVE_RIGHT: x = min(self.config.map_width - 1, x + 1)
        return (x, y)

    def _execute_non_move(self, u: PartyUnit, action: int):
        a = BossActionID(action)
        if a in (BossActionID.STAY, BossActionID.MOVE_UP, BossActionID.MOVE_DOWN,
                 BossActionID.MOVE_LEFT, BossActionID.MOVE_RIGHT):
            return

        role = u.role

        # 역할 제한
        role_allowed = {
            BossActionID.ATTACK_BASIC: True,
            BossActionID.ATTACK_SKILL: True,
            BossActionID.TAUNT: role == PartyRole.TANK,
            BossActionID.GUARD: role == PartyRole.TANK,
            BossActionID.HEAL: role == PartyRole.HEALER,
            BossActionID.CLEANSE: role == PartyRole.HEALER,
            BossActionID.BUFF_ATK: role == PartyRole.SUPPORT,
            BossActionID.BUFF_SHIELD: role == PartyRole.SUPPORT,
        }
        if not role_allowed.get(a, False):
            self.step_events[u.uid].append({"type": "invalid_action"})
            return

        # 쿨타임 체크
        if u.cooldowns.get(int(a), 0) > 0:
            self.step_events[u.uid].append({"type": "invalid_action"})
            return

        if a == BossActionID.ATTACK_BASIC:
            self._do_attack(u, skill=False)
        elif a == BossActionID.ATTACK_SKILL:
            u.cooldowns[int(a)] = self.config.skill_cooldown
            self._do_attack(u, skill=True)
        elif a == BossActionID.TAUNT:
            u.cooldowns[int(a)] = self.config.skill_cooldown
            self.boss.add_aggro(u.uid, self.config.aggro_taunt_bonus)
            # 스태거 기여
            if self.boss.stagger_active:
                self.boss.stagger_gauge -= self.config.stagger_contrib_taunt
            self.step_events[u.uid].append({"type": "taunt"})
        elif a == BossActionID.GUARD:
            u.cooldowns[int(a)] = self.config.skill_cooldown
            u.buff_guard = 1
        elif a == BossActionID.HEAL:
            u.cooldowns[int(a)] = self.config.skill_cooldown
            self._do_heal(u)
        elif a == BossActionID.CLEANSE:
            u.cooldowns[int(a)] = self.config.skill_cooldown
            self._do_cleanse(u)
        elif a == BossActionID.BUFF_ATK:
            u.cooldowns[int(a)] = self.config.skill_cooldown
            self._do_buff(u, kind="atk")
        elif a == BossActionID.BUFF_SHIELD:
            u.cooldowns[int(a)] = self.config.skill_cooldown
            self._do_buff(u, kind="shield")

    # ────────────── 액션 효과 ──────────────

    def _do_attack(self, u: PartyUnit, skill: bool):
        # 보스 사거리 체크 (보스 2x2 영역까지의 최소 체비셰프 거리)
        dist = self._boss_dist(u.x, u.y)
        if dist > u.attack_range:
            self.step_events[u.uid].append({"type": "invalid_action"})
            return
        dmg = u.attack * (2 if skill else 1)
        if u.buff_atk > 0:
            dmg = int(dmg * 1.3)
        actual = self.boss.take_damage(dmg, u.uid)
        u.total_damage_dealt += actual
        self.step_events[u.uid].append({"type": "damage", "amount": actual, "skill": skill})
        # 스태거 기여
        if self.boss.stagger_active:
            contrib = (self.config.stagger_contrib_skill if skill else self.config.stagger_contrib_basic)
            self.boss.stagger_gauge -= contrib
        # 어그로 1위 소비 (기본공격은 어그로를 "소비")
        if not skill:
            self.boss.aggro[u.uid] = max(0, self.boss.aggro.get(u.uid, 0) - self.config.aggro_basic_target_cost * 0.1)

    def _do_heal(self, u: PartyUnit):
        # 가장 HP 비율 낮은 아군에게 힐 (사거리 내)
        candidates = [
            x for x in self.units.values()
            if x.alive and abs(x.x - u.x) + abs(x.y - u.y) <= u.attack_range
        ]
        if not candidates: return
        target = min(candidates, key=lambda x: x.hp / max(1, x.max_hp))
        heal = 40
        amount = min(target.max_hp - target.hp, heal)
        target.hp += amount
        u.total_heal_done += amount
        self.step_events[u.uid].append({"type": "heal", "target": target.uid, "amount": amount})

    def _do_cleanse(self, u: PartyUnit):
        # 사거리 내 marked_turns / chain_turns 해제
        for x in self.units.values():
            if not x.alive: continue
            if abs(x.x - u.x) + abs(x.y - u.y) > u.attack_range + 1: continue
            if x.marked_turns > 0:
                x.marked_turns = 0
                self.step_events[u.uid].append({"type": "cleanse", "target": x.uid})
            # 체인 해제는 아님 — 체인은 유지 조건만 있음

    def _do_buff(self, u: PartyUnit, kind: str):
        # 가장 가까운 아군 (자신 제외)에게 버프
        candidates = [x for x in self.units.values() if x.alive and x.uid != u.uid]
        if not candidates: return
        target = min(candidates, key=lambda x: abs(x.x - u.x) + abs(x.y - u.y))
        if abs(target.x - u.x) + abs(target.y - u.y) > u.attack_range + 1: return
        if kind == "atk":
            target.buff_atk = 3
        else:
            target.buff_shield = 3
        self.step_events[u.uid].append({"type": "buff", "target": target.uid, "kind": kind})

    # ────────────── 보스 관련 ──────────────

    def _boss_tiles(self) -> Set[Pos]:
        s = set()
        for dx in range(self.config.boss_size):
            for dy in range(self.config.boss_size):
                s.add((self.boss.x + dx, self.boss.y + dy))
        return s

    def _boss_dist(self, x: int, y: int) -> int:
        """보스 2x2 영역과의 체비셰프 거리"""
        bx1, bx2 = self.boss.x, self.boss.x + self.config.boss_size - 1
        by1, by2 = self.boss.y, self.boss.y + self.config.boss_size - 1
        if bx1 <= x <= bx2 and by1 <= y <= by2:
            return 0
        dx = 0 if bx1 <= x <= bx2 else min(abs(x - bx1), abs(x - bx2))
        dy = 0 if by1 <= y <= by2 else min(abs(y - by1), abs(y - by2))
        return max(dx, dy)

    def _move_boss_toward(self, tx: int, ty: int):
        if self.boss.invuln_turns > 0 or self.boss.grog_turns > 0:
            return
        # 1칸 이동, 유닛/벽과 겹치지 않게
        unit_tiles = {(u.x, u.y) for u in self.units.values() if u.alive}
        dx = 1 if tx > self.boss.x else (-1 if tx < self.boss.x else 0)
        dy = 1 if ty > self.boss.y else (-1 if ty < self.boss.y else 0)
        for nx, ny in [(self.boss.x + dx, self.boss.y), (self.boss.x, self.boss.y + dy)]:
            # 2x2 공간 확보 가능한지
            candidate = {(nx + ddx, ny + ddy) for ddx in range(self.config.boss_size) for ddy in range(self.config.boss_size)}
            in_bounds = all(0 <= p[0] < self.config.map_width and 0 <= p[1] < self.config.map_height for p in candidate)
            if in_bounds and not (candidate & unit_tiles):
                self.boss.x, self.boss.y = nx, ny
                return

    # ────────────── 텔레그래프 Impact ──────────────

    def _apply_telegraph_impact(self, tg: ActiveTelegraph):
        pid = tg.pattern_id
        dmg = tg.extra.get("damage", 0)

        if pid == PatternID.MARK:
            # 표식 대상이 파티에서 멀리 떨어졌는지 체크
            if not tg.target_unit_ids: return
            mark_uid = tg.target_unit_ids[0]
            if mark_uid not in self.units or not self.units[mark_uid].alive: return
            mu = self.units[mark_uid]
            mu.marked_turns = 0
            # 파티 다른 유닛들과의 최소 거리
            others = [u for u in self.units.values() if u.uid != mark_uid and u.alive]
            if not others: return
            min_dist = min(max(abs(u.x - mu.x), abs(u.y - mu.y)) for u in others)
            if min_dist >= 5:
                # 파훼 성공: 보스 1턴 경직
                self.boss.grog_turns = max(self.boss.grog_turns, 1)
                for u in self.units.values():
                    self.step_events[u.uid].append({"type": "mechanic_success", "pattern": int(pid)})
            else:
                # 실패: 5x5 폭발
                for u in self.units.values():
                    if not u.alive: continue
                    if max(abs(u.x - mu.x), abs(u.y - mu.y)) <= 2:
                        self._deal_damage_to_unit(u, dmg)
                for u in self.units.values():
                    self.step_events[u.uid].append({"type": "mechanic_fail", "pattern": int(pid)})

        elif pid == PatternID.STAGGER:
            if self.boss.stagger_gauge <= 0:
                self.boss.grog_turns = self.config.stagger_success_grog_turns
                for u in self.units.values():
                    self.step_events[u.uid].append({"type": "stagger_success"})
            else:
                for u in self.units.values():
                    if u.alive:
                        self._deal_damage_to_unit(u, self.config.stagger_fail_damage)
                    self.step_events[u.uid].append({"type": "stagger_fail"})
            self.boss.stagger_active = False
            self.boss.stagger_gauge = 0

        elif pid == PatternID.CROSS_INFERNO:
            safe_quads = tg.target_unit_ids  # 2개 사분면
            for u in self.units.values():
                if not u.alive: continue
                q = self._quadrant(u.x, u.y)
                if q not in safe_quads:
                    self._deal_damage_to_unit(u, dmg)

        elif pid == PatternID.CURSED_CHAIN:
            # 시작 시점: 두 유닛에 체인 부여, 6턴 지속
            if len(tg.target_unit_ids) >= 2:
                a_uid, b_uid = tg.target_unit_ids[:2]
                if a_uid in self.units and b_uid in self.units:
                    self.units[a_uid].chained_with = b_uid
                    self.units[b_uid].chained_with = a_uid
                    self.units[a_uid].chain_turns = 6
                    self.units[b_uid].chain_turns = 6
                    self.boss.active_chain = {"pair": (a_uid, b_uid), "turns": 6}

        else:
            # 일반 장판형 패턴
            for u in self.units.values():
                if not u.alive: continue
                if (u.x, u.y) in tg.danger_tiles:
                    self._deal_damage_to_unit(u, dmg)

    def _deal_damage_to_unit(self, u: PartyUnit, amount: int):
        actual = max(1, amount - u.defense)
        if u.buff_guard > 0: actual = actual // 2
        if u.buff_shield > 0: actual = int(actual * 0.7)
        u.hp -= actual
        u.total_damage_taken += actual
        if u.hp <= 0:
            u.hp = 0
            u.alive = False
            self.step_events[u.uid].append({"type": "death"})
        else:
            self.step_events[u.uid].append({"type": "damage_taken", "amount": actual})

    def _quadrant(self, x: int, y: int) -> int:
        cx, cy = self.config.map_width // 2, self.config.map_height // 2
        if x < cx and y < cy: return 0
        if x >= cx and y < cy: return 1
        if x < cx and y >= cy: return 2
        return 3

    def _tick_chains(self):
        pair = None
        for u in self.units.values():
            if u.chain_turns > 0 and u.chained_with is not None:
                partner = self.units.get(u.chained_with)
                if partner and partner.alive:
                    d = max(abs(u.x - partner.x), abs(u.y - partner.y))
                    if d > 3:
                        self._deal_damage_to_unit(u, 25)
                u.chain_turns -= 1
                if u.chain_turns <= 0:
                    u.chained_with = None

    # ────────────── 관찰 ──────────────

    def _get_all_observations(self) -> Dict[str, np.ndarray]:
        return {aid: self._observe(self.uid_of(aid)) for aid in self.agent_ids()}

    def _observe(self, uid: int) -> np.ndarray:
        """92차원 관찰 벡터 (boss_design.md 참조)"""
        u = self.units[uid]
        w, h = self.config.map_width, self.config.map_height
        v = []

        # 자기(8): hp비율, mp비율, x정규화, y정규화, 역할 one-hot(4)
        v += [u.hp / max(1, u.max_hp), u.mp / max(1, u.max_mp),
              u.x / w, u.y / h]
        role_oh = [0, 0, 0, 0]
        role_oh[int(u.role)] = 1
        v += role_oh

        # 아군(24): 3명 × 8
        others = [ux for ux in self.units.values() if ux.uid != uid]
        for i in range(3):
            if i < len(others):
                a = others[i]
                v += [a.hp / max(1, a.max_hp), a.mp / max(1, a.max_mp),
                      a.x / w, a.y / h]
                roh = [0, 0, 0, 0]; roh[int(a.role)] = 1
                v += roh
            else:
                v += [0.0] * 8

        # 보스(10): hp비율, x, y, 페이즈(3), 어그로1위 one-hot(4)
        v += [self.boss.hp / self.config.boss_max_hp,
              self.boss.x / w, self.boss.y / h]
        ph = [0, 0, 0]; ph[int(self.boss.phase)] = 1
        v += ph
        top = self.boss.top_aggro_uid() or 0
        aoh = [0, 0, 0, 0]; aoh[top] = 1
        v += aoh

        # 어그로 비율(4)
        total = sum(self.boss.aggro.values()) + 1e-6
        v += [self.boss.aggro.get(i, 0) / total for i in range(4)]

        # 텔레그래프(24): 최대 2개 × (패턴 one-hot(8) + 남은턴 정규화(1) + 대상 one-hot(3))
        for i in range(2):
            if i < len(self.boss.telegraphs):
                tg = self.boss.telegraphs[i]
                poh = [0] * 8; poh[int(tg.pattern_id)] = 1
                v += poh
                v += [tg.turns_remaining / max(1, tg.total_wind_up)]
                toh = [0, 0, 0]
                # 대상 uid가 자기면 0, 다른 아군 중 첫번째면 1 등 축약
                if tg.target_unit_ids and tg.target_unit_ids[0] == uid:
                    toh[0] = 1
                elif tg.target_unit_ids:
                    toh[1] = 1
                else:
                    toh[2] = 1
                v += toh
            else:
                v += [0.0] * 12

        # 위험 타일 3x3 (9): 자기 위치 ±1
        danger_tiles = set()
        for tg in self.boss.telegraphs:
            danger_tiles |= tg.danger_tiles
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                p = (u.x + dx, u.y + dy)
                v += [1.0 if p in danger_tiles else 0.0]

        # 스태거(2)
        v += [self.boss.stagger_gauge / max(1, self.config.stagger_gauge),
              1.0 if self.boss.stagger_active else 0.0]

        # 저주 연결(3): 활성, 파트너 존재, 파트너 거리
        chain_active = 1.0 if u.chained_with is not None else 0.0
        partner_dist = 0.0
        if u.chained_with is not None and u.chained_with in self.units:
            p = self.units[u.chained_with]
            partner_dist = max(abs(u.x - p.x), abs(u.y - p.y)) / max(w, h)
        v += [chain_active, 1.0 if chain_active > 0 else 0.0, partner_dist]

        # 안전 지대 방향(4): 현재 P7 활성 시 2개 사분면 표시
        safe_dir = [0.0, 0.0, 0.0, 0.0]
        for tg in self.boss.telegraphs:
            if tg.pattern_id == PatternID.CROSS_INFERNO:
                for q in tg.target_unit_ids:
                    safe_dir[q] = 1.0
                break
        v += safe_dir

        # 플레이어 정보(4): 위치, HP, 거리
        player = self.units[self.config.player_slot]
        v += [player.x / w, player.y / h,
              player.hp / max(1, player.max_hp),
              (abs(u.x - player.x) + abs(u.y - player.y)) / (w + h)]

        arr = np.array(v, dtype=np.float32)
        # 크기 맞추기 — 설계 92차원
        expected = self.config.obs_size
        if arr.shape[0] < expected:
            arr = np.pad(arr, (0, expected - arr.shape[0]))
        elif arr.shape[0] > expected:
            arr = arr[:expected]
        return arr

    # ────────────── info ──────────────

    def _build_infos(self) -> Dict[str, dict]:
        return {
            aid: {
                "events": self.step_events.get(self.uid_of(aid), []),
                "victory": self.victory,
                "wipe": self.wipe,
                "step": self.current_step,
                "boss_hp_ratio": self.boss.hp / self.config.boss_max_hp,
                "phase": int(self.boss.phase),
            }
            for aid in self.agent_ids()
        }

    # ────────────── 스냅샷 (Unity 전송용) ──────────────

    def get_snapshot(self) -> dict:
        """Unity 스트리머에서 JSON으로 직렬화해 보낼 게임 상태"""
        return {
            "step": self.current_step,
            "boss": {
                "x": self.boss.x, "y": self.boss.y,
                "hp": self.boss.hp, "max_hp": self.config.boss_max_hp,
                "phase": int(self.boss.phase),
                "invuln": self.boss.invuln_turns, "grog": self.boss.grog_turns,
                "stagger_active": self.boss.stagger_active,
                "stagger_gauge": self.boss.stagger_gauge,
            },
            "units": [
                {
                    "uid": u.uid, "role": int(u.role),
                    "x": u.x, "y": u.y,
                    "hp": u.hp, "max_hp": u.max_hp,
                    "alive": u.alive,
                    "marked": u.marked_turns > 0,
                    "chained_with": u.chained_with,
                    "buff_atk": u.buff_atk, "buff_shield": u.buff_shield,
                }
                for u in self.units.values()
            ],
            "telegraphs": [
                {
                    "pattern": int(tg.pattern_id),
                    "turns_remaining": tg.turns_remaining,
                    "total_wind_up": tg.total_wind_up,
                    "danger_tiles": list(tg.danger_tiles),
                    "target_uids": list(tg.target_unit_ids),
                }
                for tg in self.boss.telegraphs
            ],
            "done": self.done,
            "victory": self.victory,
            "wipe": self.wipe,
        }
