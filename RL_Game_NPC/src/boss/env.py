"""BossRaidEnv — 유클리드 연속 공간 1보스 4파티 레이드 환경

- 위치: float (0.0 ~ map_width/height)
- 이동: 4방향, 한 턴당 move_speed 만큼 (유닛 반경 기반 충돌)
- 패턴: 기하 도형 (Circle/Line/Fan/Cross) — contains(pos) 판정
- 관찰: 상대 위치 벡터 + 8방향 위험 센서
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
import random

from .config import (
    BossConfig, PartyRole, PatternID, PhaseID, BossActionID,
    ROLE_STATS_BOSS,
)
from .boss import Boss
from .patterns import ActiveTelegraph, Pos, sample_danger_sensor
from .rewards import RewardComputer


# ─────────────────── 파티 유닛 (유클리드) ───────────────────

@dataclass
class PartyUnit:
    uid: int
    role: PartyRole
    x: float
    y: float
    hp: int
    mp: int
    max_hp: int
    max_mp: int
    attack: int
    defense: int
    attack_range: float
    move_speed: float
    radius: float
    alive: bool = True
    cooldowns: Dict[int, int] = field(default_factory=dict)
    buff_atk: int = 0
    buff_shield: int = 0
    buff_guard: int = 0
    marked_turns: int = 0
    chained_with: Optional[int] = None
    chain_turns: int = 0
    total_damage_dealt: int = 0
    total_damage_taken: int = 0
    total_heal_done: int = 0


# ─────────────────── 환경 ───────────────────

class BossRaidEnv:
    """보스 레이드 환경 (유클리드).

    agent_ids: "p0" ~ "p3" (DEALER, TANK, HEALER, SUPPORT)
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
        self.combat_started = False  # user_test_mode: 플레이어가 보스 인식 범위 진입 전엔 False

        self.reward_computer = RewardComputer(self.config)
        self.step_events: Dict[int, List[dict]] = {}

        # 속도 계산용 이전 위치
        self._prev_unit_positions: Dict[int, Tuple[float, float]] = {}
        self._prev_boss_pos: Tuple[float, float] = (0.0, 0.0)

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
        self.combat_started = False
        self.step_events.clear()
        self._seal_hold_count = 0

        self.units.clear()

        # ── 보스 위치 ──
        if self.config.user_test_mode:
            # 실전 시연: 보스 고정 spawn (우상단 등 정해진 위치)
            boss_x = self.config.boss_spawn_x
            boss_y = self.config.boss_spawn_y
            # 파티는 맵 반대편 (좌하단) 에서 시작 → 플레이어가 보스 쪽으로 이동해야 함
            margin = 3.0
            center_x = max(margin, min(self.config.map_width - margin, self.config.map_width - boss_x))
            center_y = max(margin, min(self.config.map_height - margin, self.config.map_height - boss_y))
        else:
            # 학습용 랜덤 배치
            margin = 3.0
            boss_x = self.rng.uniform(margin, self.config.map_width - margin)
            boss_y = self.rng.uniform(margin, self.config.map_height - margin)
            party_angle = self.rng.uniform(0, 2 * math.pi)
            party_dist = self.rng.uniform(5.0, 8.0)
            center_x = boss_x + math.cos(party_angle) * party_dist
            center_y = boss_y + math.sin(party_angle) * party_dist
            center_x = max(margin, min(self.config.map_width - margin, center_x))
            center_y = max(margin, min(self.config.map_height - margin, center_y))

        # 4명을 중심 주변 ±1.0 범위에 흩뿌림
        for i, role in enumerate(self.config.party_roles):
            stats = ROLE_STATS_BOSS[role]
            sx = center_x + self.rng.uniform(-1.0, 1.0)
            sy = center_y + self.rng.uniform(-1.0, 1.0)
            sx = max(stats.radius, min(self.config.map_width - stats.radius, sx))
            sy = max(stats.radius, min(self.config.map_height - stats.radius, sy))
            u = PartyUnit(
                uid=i, role=role, x=sx, y=sy,
                hp=stats.hp, mp=stats.mp,
                max_hp=stats.hp, max_mp=stats.mp,
                attack=stats.attack, defense=stats.defense,
                attack_range=stats.attack_range,
                move_speed=stats.move_speed,
                radius=stats.radius,
            )
            self.units[i] = u

        self.boss = Boss(config=self.config, rng=self.rng)
        self.boss.x = boss_x
        self.boss.y = boss_y
        self.boss.hp = self.config.boss_max_hp

        # 초기 어그로 약간 랜덤화 (탱커 고정 학습 방지)
        for u in self.units.values():
            self.boss.aggro[u.uid] = self.rng.uniform(0, 5)

        self._prev_boss_pos = (self.boss.x, self.boss.y)
        self._prev_unit_positions = {u.uid: (u.x, u.y) for u in self.units.values()}

        return self._get_all_observations()

    # ────────────── step ──────────────

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        self._prev_boss_pos = (self.boss.x, self.boss.y)
        self._prev_unit_positions = {u.uid: (u.x, u.y) for u in self.units.values()}

        self.step_events = {uid: [] for uid in self.units}
        self.current_step += 1

        # ── 0. 실전 모드: 전투 시작 게이트 ──
        if self.config.user_test_mode and not self.combat_started:
            player = self.units[self.config.player_slot]
            d = math.hypot(player.x - self.boss.x, player.y - self.boss.y)
            if d <= self.config.boss_detection_range:
                self.combat_started = True
            else:
                # 전투 미시작: 파티 이동만 처리, 보스 정지·패턴 비활성
                self._resolve_party_actions(actions)
                # 버프 디버프 틱은 진행 (쿨다운만)
                for u in self.units.values():
                    for k in list(u.cooldowns.keys()):
                        u.cooldowns[k] = max(0, u.cooldowns[k] - 1)
                obs = self._get_all_observations()
                rewards = {aid: 0.0 for aid in self.agent_ids()}
                dones = {aid: False for aid in self.agent_ids()}
                infos = self._build_infos()
                return obs, rewards, dones, infos

        # 1. 페이즈 전이 → Seal Break 자동 발동
        phase_changed = self.boss.check_phase_transition()
        if phase_changed:
            for uid in self.units:
                self.step_events[uid].append({"type": "phase_clear"})
            # 봉인 해제 패턴 강제 발동
            self.boss.start_pattern(
                PatternID.SEAL_BREAK,
                self.party_positions(),
                self.party_roles(),
                extra_override={"dealer_uid": self.config.player_slot},
            )

        # 2. 텔레그래프 카운트다운
        ready_telegraphs = self.boss.tick_telegraphs()

        # 3. 파티 행동 (동시 처리)
        self._resolve_party_actions(actions)

        # 4. 텔레그래프 impact
        for tg in ready_telegraphs:
            self._apply_telegraph_impact(tg)
            tg.impacted = True

        # 5. Seal Break 활성 중이면 보스 정지 (이동/패턴 없음)
        seal_active = any(tg.pattern_id == PatternID.SEAL_BREAK for tg in self.boss.telegraphs)

        if not seal_active and not self.boss.stagger_active:
            # 신규 패턴 시전 (스태거 wind_up 중엔 다른 패턴 차단 — 파티 집결 보장)
            if self.boss.invuln_turns <= 0 and self.boss.grog_turns <= 0:
                if len(self.boss.telegraphs) < self.config.max_active_telegraphs:
                    pid = self.boss.select_pattern()
                    if pid is not None:
                        self.boss.start_pattern(pid, self.party_positions(), self.party_roles())

        # 5b. Seal Break 매 턴 유지 체크
        self._tick_seal_break()

        # 6. 체인 지속 효과
        self._tick_chains()

        # 7. 보스 이동 — Seal 중이면 정지
        if not seal_active:
            top_uid = self.boss.top_aggro_uid()
            if top_uid is not None and top_uid in self.units and self.units[top_uid].alive:
                t = self.units[top_uid]
                others = [(u.x, u.y, u.radius) for u in self.units.values()
                          if u.alive and u.uid != top_uid]
                self.boss.move_toward(t.x, t.y, others)

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
        # NOTE: 학습 중 딜러(플레이어) 사망해도 에피소드 계속 (NPC 학습 기회 보장)
        #       실전(boss_streamer)에서는 wipe에 딜러 포함되므로 문제 없음
        if self.config.user_test_mode:
            # 실전 모드: 보스 처치 또는 전멸까지 계속 (max_steps 무시)
            self.done = self.victory or self.wipe
        else:
            self.done = (
                self.victory or self.wipe or
                self.current_step >= self.config.max_steps
            )

        obs = self._get_all_observations()
        rewards = self.reward_computer.compute(self)
        dones = {aid: self.done for aid in self.agent_ids()}
        infos = self._build_infos()
        return obs, rewards, dones, infos

    # ────────────── 파티 행동 ──────────────

    def _resolve_party_actions(self, actions: Dict[str, int]):
        # 이동 의도 수집
        move_intents: Dict[int, Tuple[float, float]] = {}
        for aid, action in actions.items():
            uid = self.uid_of(aid)
            u = self.units[uid]
            if not u.alive: continue
            move_intents[uid] = self._compute_move_delta(u, action)

        # 동시 이동 (간단 충돌: uid 낮은 순으로 자리 잡음)
        for uid in sorted(move_intents.keys()):
            dx, dy = move_intents[uid]
            if dx == 0 and dy == 0: continue
            u = self.units[uid]
            new_x = u.x + dx; new_y = u.y + dy
            # 맵 경계
            new_x = max(u.radius, min(self.config.map_width - u.radius, new_x))
            new_y = max(u.radius, min(self.config.map_height - u.radius, new_y))
            # 다른 유닛/보스 충돌 체크
            blocked = False
            for ox, oy, orad in self._occupied_circles(exclude_uid=uid):
                if math.hypot(new_x - ox, new_y - oy) < u.radius + orad - 0.05:
                    blocked = True; break
            if not blocked:
                u.x, u.y = new_x, new_y

        # 비이동 액션
        for aid, action in actions.items():
            uid = self.uid_of(aid)
            u = self.units[uid]
            if not u.alive: continue
            self._execute_non_move(u, action)

    def _compute_move_delta(self, u: PartyUnit, action: int) -> Tuple[float, float]:
        s = u.move_speed
        d = s * 0.7071  # 대각 이동: 같은 거리 유지 (s / sqrt(2))
        if action == BossActionID.MOVE_UP:         return (0.0, -s)
        if action == BossActionID.MOVE_DOWN:       return (0.0, s)
        if action == BossActionID.MOVE_LEFT:       return (-s, 0.0)
        if action == BossActionID.MOVE_RIGHT:      return (s, 0.0)
        if action == BossActionID.MOVE_UP_LEFT:    return (-d, -d)
        if action == BossActionID.MOVE_UP_RIGHT:   return (d, -d)
        if action == BossActionID.MOVE_DOWN_LEFT:  return (-d, d)
        if action == BossActionID.MOVE_DOWN_RIGHT: return (d, d)
        return (0.0, 0.0)

    def _occupied_circles(self, exclude_uid: Optional[int] = None):
        """현재 맵의 점유 원형 리스트 반환: (x, y, radius).

        MMORPG 관례: 파티원끼리는 겹쳐도 됨 (stagger 집결 등 필요). 보스만 몸통 차단.
        """
        return [(self.boss.x, self.boss.y, self.config.boss_radius)]

    def _execute_non_move(self, u: PartyUnit, action: int):
        a = BossActionID(action)
        if a in (BossActionID.STAY,
                 BossActionID.MOVE_UP, BossActionID.MOVE_DOWN,
                 BossActionID.MOVE_LEFT, BossActionID.MOVE_RIGHT,
                 BossActionID.MOVE_UP_LEFT, BossActionID.MOVE_UP_RIGHT,
                 BossActionID.MOVE_DOWN_LEFT, BossActionID.MOVE_DOWN_RIGHT):
            return

        role = u.role
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
            if self.boss.stagger_active:
                self.boss.stagger_gauge -= self.config.stagger_contrib_taunt
                self.step_events[u.uid].append({
                    "type": "stagger_contribute",
                    "amount": self.config.stagger_contrib_taunt,
                })
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

    # ────────────── 액션 효과 (유클리드 거리) ──────────────

    def _do_attack(self, u: PartyUnit, skill: bool):
        # Seal Break 중 보스 데미지 50% 감소 (완전 무적이면 학습 불가)
        seal_active = any(tg.pattern_id == PatternID.SEAL_BREAK for tg in self.boss.telegraphs)

        dist = self._boss_dist(u.x, u.y)
        if dist > u.attack_range:
            self.step_events[u.uid].append({"type": "invalid_action"})
            return
        dmg = u.attack * (2 if skill else 1)
        if seal_active:
            dmg = dmg // 2                               # Seal 중 딜 50% 감소
        if u.buff_atk > 0:
            dmg = int(dmg * 1.3)
        actual = self.boss.take_damage(dmg, u.uid)
        u.total_damage_dealt += actual
        self.step_events[u.uid].append({"type": "damage", "amount": actual, "skill": skill})
        if self.boss.stagger_active:
            contrib = (self.config.stagger_contrib_skill if skill else self.config.stagger_contrib_basic)
            self.boss.stagger_gauge -= contrib
            self.step_events[u.uid].append({
                "type": "stagger_contribute", "amount": contrib,
            })
        if not skill:
            self.boss.aggro[u.uid] = max(0, self.boss.aggro.get(u.uid, 0)
                                           - self.config.aggro_basic_target_cost * 0.1)

    def _do_heal(self, u: PartyUnit):
        candidates = [
            x for x in self.units.values()
            if x.alive and math.hypot(x.x - u.x, x.y - u.y) <= u.attack_range
        ]
        if not candidates: return
        target = min(candidates, key=lambda x: x.hp / max(1, x.max_hp))
        heal = 80                                 # 60 → 80 (plateau 탈출용 생존력 ↑↑)
        amount = min(target.max_hp - target.hp, heal)
        target.hp += amount
        u.total_heal_done += amount
        self.step_events[u.uid].append({"type": "heal", "target": target.uid, "amount": amount})

    def _do_cleanse(self, u: PartyUnit):
        for x in self.units.values():
            if not x.alive: continue
            if math.hypot(x.x - u.x, x.y - u.y) > u.attack_range + 0.5: continue
            if x.marked_turns > 0:
                x.marked_turns = 0
                self.step_events[u.uid].append({"type": "cleanse", "target": x.uid})

    def _do_buff(self, u: PartyUnit, kind: str):
        candidates = [x for x in self.units.values() if x.alive and x.uid != u.uid]
        if not candidates: return
        target = min(candidates, key=lambda x: math.hypot(x.x - u.x, x.y - u.y))
        if math.hypot(target.x - u.x, target.y - u.y) > u.attack_range + 0.5: return
        if kind == "atk":
            target.buff_atk = 3
        else:
            target.buff_shield = 3
        self.step_events[u.uid].append({"type": "buff", "target": target.uid, "kind": kind})

    # ────────────── 보스 관련 ──────────────

    def _boss_dist(self, x: float, y: float) -> float:
        """유닛 위치에서 보스 원형 표면까지의 유클리드 거리 (음수면 안에 있음)."""
        center_dist = math.hypot(x - self.boss.x, y - self.boss.y)
        return max(0.0, center_dist - self.config.boss_radius)

    def _quadrant(self, x: float, y: float) -> int:
        cx, cy = self.config.map_width / 2.0, self.config.map_height / 2.0
        q = 0
        if x >= cx: q |= 1
        if y >= cy: q |= 2
        return q

    # ────────────── 텔레그래프 Impact ──────────────

    def _apply_telegraph_impact(self, tg: ActiveTelegraph):
        pid = tg.pattern_id
        dmg = tg.extra.get("damage", 0)

        if pid == PatternID.MARK:
            if not tg.target_unit_ids: return
            mark_uid = tg.target_unit_ids[0]
            if mark_uid not in self.units or not self.units[mark_uid].alive: return
            mu = self.units[mark_uid]
            mu.marked_turns = 0
            # 파훼 성공 조건: 다른 유닛 모두가 escape_distance 이상 떨어짐
            others = [u for u in self.units.values() if u.uid != mark_uid and u.alive]
            if not others: return
            min_dist = min(math.hypot(u.x - mu.x, u.y - mu.y) for u in others)
            if min_dist >= self.config.pat_mark_escape_distance:
                self.boss.grog_turns = max(self.boss.grog_turns, 1)
                for u in self.units.values():
                    self.step_events[u.uid].append({"type": "mechanic_success", "pattern": int(pid)})
            else:
                # 표식 위치 기준 원형 폭발
                for u in self.units.values():
                    if not u.alive: continue
                    if math.hypot(u.x - mu.x, u.y - mu.y) <= self.config.pat_mark_blast_radius:
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

        elif pid == PatternID.CURSED_CHAIN:
            if len(tg.target_unit_ids) >= 2:
                a_uid, b_uid = tg.target_unit_ids[:2]
                if a_uid in self.units and b_uid in self.units:
                    self.units[a_uid].chained_with = b_uid
                    self.units[b_uid].chained_with = a_uid
                    self.units[a_uid].chain_turns = 6
                    self.units[b_uid].chain_turns = 6
                    self.boss.active_chain = {"pair": (a_uid, b_uid), "turns": 6}

        elif pid == PatternID.SEAL_BREAK:
            # Seal Break는 _tick_seal_break()에서 매 턴 처리.
            # wind_up 만료 시 여기 도달하면 이미 처리됐거나 타임아웃.
            pass

        else:
            # 일반 기하 패턴 — tg.contains() 사용
            for u in self.units.values():
                if not u.alive: continue
                if tg.contains((u.x, u.y)):
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

    # ────────────── Seal Break 매 턴 체크 ──────────────

    _seal_hold_count: int = 0   # 연속 유지 턴 수

    def _tick_seal_break(self):
        """Seal Break: 매 턴 3장판 점유 상태 체크.
        - 도착 단계: 아직 3개 안 채움 → 카운트 안 함
        - 유지 단계: 3개 모두 채워진 순간부터 hold 카운트
        - hold 중 누가 나가면 → 즉시 전멸
        - hold 완료 → 성공
        """
        from itertools import permutations
        for tg in self.boss.telegraphs:
            if tg.pattern_id != PatternID.SEAL_BREAK:
                continue

            # ── 동적 재배정: 딜러가 향한 spot 을 dealer_spot 으로, NPC 는 나머지 ──
            dealer = self.units[self.config.player_slot]
            dealer_spot_idx = min(range(len(tg.shapes)),
                                  key=lambda i: math.hypot(
                                      dealer.x - tg.shapes[i].params["cx"],
                                      dealer.y - tg.shapes[i].params["cy"]
                                  ))
            if tg.target_unit_ids:
                tg.target_unit_ids[0] = dealer_spot_idx
            available = [i for i in range(len(tg.shapes)) if i != dealer_spot_idx]
            npc_uids = sorted(uid for uid in self.units
                              if uid != self.config.player_slot
                              and self.units[uid].alive)
            if npc_uids:
                best_perm, best_total = None, 1e18
                for perm in permutations(available, len(npc_uids)):
                    total = sum(math.hypot(
                        self.units[uid].x - tg.shapes[si].params["cx"],
                        self.units[uid].y - tg.shapes[si].params["cy"]
                    ) for uid, si in zip(npc_uids, perm))
                    if total < best_total:
                        best_total, best_perm = total, perm
                if best_perm:
                    tg.extra["npc_spots"] = dict(zip(npc_uids, best_perm))

            npc_spots = []

            for si, shape in enumerate(tg.shapes):
                if si == dealer_spot_idx:
                    continue
                sx, sy = shape.params["cx"], shape.params["cy"]
                occupied = False
                for u in self.units.values():
                    if u.uid == self.config.player_slot or not u.alive:
                        continue
                    if math.hypot(u.x - sx, u.y - sy) <= shape.params["r"]:
                        occupied = True
                        break
                npc_spots.append(occupied)

            all_occupied = all(npc_spots) and len(npc_spots) >= 3

            if all_occupied:
                self._seal_hold_count += 1

                # 유지 중 보상 (매 턴)
                for u in self.units.values():
                    self.step_events[u.uid].append({
                        "type": "seal_holding",
                        "hold": self._seal_hold_count,
                        "needed": self.config.pat_seal_hold_turns,
                    })

                # 유지 완료!
                if self._seal_hold_count >= self.config.pat_seal_hold_turns:
                    self.boss.grog_turns = max(self.boss.grog_turns, self.config.pat_seal_success_grog)
                    for u in self.units.values():
                        self.step_events[u.uid].append({"type": "seal_success"})
                    # 텔레그래프 제거
                    tg.impacted = True
                    tg.post_impact_turns = 0
                    self._seal_hold_count = 0
                    return

            elif self._seal_hold_count > 0:
                # 유지 중 누가 나감 → 즉시 전멸!
                for u in self.units.values():
                    if u.alive:
                        self._deal_damage_to_unit(u, self.config.pat_seal_fail_damage)
                    self.step_events[u.uid].append({"type": "seal_fail", "reason": "broke_hold"})
                tg.impacted = True
                tg.post_impact_turns = 0
                self._seal_hold_count = 0
                return

            # 도착 시간 초과 체크 (wind_up 끝났는데 아직 hold 시작 안 함)
            arrive_deadline = tg.total_wind_up - self.config.pat_seal_hold_turns
            turns_elapsed = tg.total_wind_up - tg.turns_remaining
            if turns_elapsed >= arrive_deadline and self._seal_hold_count == 0 and not all_occupied:
                # 도착 시간 지남 + 아직 3명 안 모임 → 실패
                for u in self.units.values():
                    if u.alive:
                        self._deal_damage_to_unit(u, self.config.pat_seal_fail_damage)
                    self.step_events[u.uid].append({"type": "seal_fail", "reason": "timeout"})
                tg.impacted = True
                tg.post_impact_turns = 0
                self._seal_hold_count = 0
                return

            return  # Seal은 최대 1개

    def _tick_chains(self):
        for u in self.units.values():
            if u.chain_turns > 0 and u.chained_with is not None:
                partner = self.units.get(u.chained_with)
                if partner and partner.alive:
                    d = math.hypot(u.x - partner.x, u.y - partner.y)
                    if d > self.config.pat_chain_max_distance:
                        self._deal_damage_to_unit(u, self.config.chain_damage)
                u.chain_turns -= 1
                if u.chain_turns <= 0:
                    u.chained_with = None

    # ────────────── 관찰 (유클리드) ──────────────

    def _get_all_observations(self) -> Dict[str, np.ndarray]:
        return {aid: self._observe(self.uid_of(aid)) for aid in self.agent_ids()}

    def _observe(self, uid: int) -> np.ndarray:
        """B안 관측 벡터 (123차원).

        블록:
          Self(15)        : hp/mp/x/y + role(OH4) + radius + cd_skill + cd_role_a + cd_role_b
                          + step_progress + aggro_ratio + is_top_aggro
          Allies(24)      : 3명 × 8 = dx/dy(/10) + hp + alive + role(OH4)
          Boss(10)        : dx/dy/dist(/10) + hp + phase(OH3) + grog + invuln + facing
          PatternCh(45)   : 9패턴 × 5 = active + turns_norm + am_I_target + target_dx/dy(/10)
          DangerSensor(8) : 8방향 거리 (1=안전)
          Escape(4)       : in_danger + escape_dx + escape_dy + urgency
          Coop(13)        : Chain(4) + Cross(3) + Stagger(2) + Seal(4)
          Player(4)       : dx/dy/dist(/10) + hp
        합계: 15+24+10+45+8+4+13+4 = 123
        """
        u = self.units[uid]
        cfg = self.config
        v: List[float] = []

        top_uid = self.boss.top_aggro_uid()

        # ── [1] Self (15) ──
        v += [
            u.hp / max(1, u.max_hp),
            u.mp / max(1, u.max_mp),
            u.x / cfg.map_width,
            u.y / cfg.map_height,
        ]
        role_oh = [0.0] * 4; role_oh[int(u.role)] = 1.0
        v += role_oh
        v.append(u.radius)
        cd_max = float(max(1, cfg.skill_cooldown))
        v.append(u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) / cd_max)
        role_skills = {
            PartyRole.DEALER:  (int(BossActionID.ATTACK_SKILL), int(BossActionID.ATTACK_SKILL)),
            PartyRole.TANK:    (int(BossActionID.TAUNT), int(BossActionID.GUARD)),
            PartyRole.HEALER:  (int(BossActionID.HEAL), int(BossActionID.CLEANSE)),
            PartyRole.SUPPORT: (int(BossActionID.BUFF_ATK), int(BossActionID.BUFF_SHIELD)),
        }
        skill_a, skill_b = role_skills[u.role]
        v.append(u.cooldowns.get(skill_a, 0) / cd_max)
        v.append(u.cooldowns.get(skill_b, 0) / cd_max)
        v.append(self.current_step / max(1, cfg.max_steps))
        total_aggro = sum(self.boss.aggro.values()) + 1e-6
        v.append(self.boss.aggro.get(uid, 0.0) / total_aggro)
        v.append(1.0 if top_uid == uid else 0.0)

        # ── [2] Allies (24) — 자기 제외 3명, uid 오름차순 고정 슬롯 ──
        allies = sorted((a for a in self.units.values() if a.uid != uid),
                        key=lambda a: a.uid)
        for i in range(3):
            if i < len(allies):
                a = allies[i]
                v += [
                    (a.x - u.x) / 10.0,
                    (a.y - u.y) / 10.0,
                    a.hp / max(1, a.max_hp),
                    1.0 if a.alive else 0.0,
                ]
                a_oh = [0.0] * 4; a_oh[int(a.role)] = 1.0
                v += a_oh
            else:
                v += [0.0] * 8

        # ── [3] Boss (10) ──
        bdx = self.boss.x - u.x; bdy = self.boss.y - u.y
        bdist = math.hypot(bdx, bdy)
        v += [bdx / 10.0, bdy / 10.0, bdist / 10.0,
              self.boss.hp / cfg.boss_max_hp]
        ph_oh = [0.0] * 3; ph_oh[int(self.boss.phase)] = 1.0
        v += ph_oh
        v += [1.0 if self.boss.grog_turns > 0 else 0.0,
              1.0 if self.boss.invuln_turns > 0 else 0.0]
        facing = 0.0
        if top_uid is not None and top_uid in self.units:
            t = self.units[top_uid]
            facing = math.atan2(t.y - self.boss.y, t.x - self.boss.x) / math.pi
        v.append(facing)

        # ── [4] Pattern channels (45) = 9 패턴 × 5 (패턴ID=인덱스 고정) ──
        tg_by_pattern = {int(tg.pattern_id): tg for tg in self.boss.telegraphs}
        for pid_int in range(9):
            tg = tg_by_pattern.get(pid_int)
            if tg is None:
                v += [0.0, 0.0, 0.0, 0.0, 0.0]
                continue
            turns_norm = tg.turns_remaining / max(1, tg.total_wind_up)
            am_I, tdx, tdy = self._pattern_target_info(tg, PatternID(pid_int), u, uid)
            v += [1.0, turns_norm, am_I, tdx, tdy]

        # ── [5] Danger sensor (8) ──
        v += sample_danger_sensor((u.x, u.y), self.boss.telegraphs)

        # ── [6] Escape guide (4) ──
        in_danger = 0.0
        escape_dx, escape_dy, urgency = 0.0, 0.0, 0.0
        danger_tgs = [tg for tg in self.boss.telegraphs
                      if tg.pattern_id != PatternID.SEAL_BREAK
                      and tg.contains((u.x, u.y))]
        if danger_tgs:
            in_danger = 1.0
            urgent_tg = min(danger_tgs, key=lambda t: t.turns_remaining)
            urgency = 1.0 - urgent_tg.turns_remaining / max(1, urgent_tg.total_wind_up)
            best_step = 1e9
            for di in range(8):
                theta = di * math.pi / 4
                for step_m in (0.5, 1.0, 1.5, 2.0, 3.0):
                    tx = u.x + math.cos(theta) * step_m
                    ty = u.y + math.sin(theta) * step_m
                    if not urgent_tg.contains((tx, ty)):
                        if step_m < best_step:
                            best_step = step_m
                            escape_dx = math.cos(theta)
                            escape_dy = math.sin(theta)
                        break
        v += [in_danger, escape_dx, escape_dy, urgency]

        # ── [7] Coop context (13) ──
        # Chain (4): wind_up/post-impact 통합
        chain_partner_active = 0.0
        chain_pdx, chain_pdy, chain_slack = 0.0, 0.0, 0.0
        partner_uid = None
        if u.chained_with is not None and u.chained_with in self.units:
            partner_uid = u.chained_with
        else:
            for tg in self.boss.telegraphs:
                if tg.pattern_id == PatternID.CURSED_CHAIN and len(tg.target_unit_ids) >= 2:
                    a, b = tg.target_unit_ids[:2]
                    if uid == a:   partner_uid = b; break
                    if uid == b:   partner_uid = a; break
        if partner_uid is not None and partner_uid in self.units:
            p = self.units[partner_uid]
            chain_partner_active = 1.0
            chain_pdx = (p.x - u.x) / 10.0
            chain_pdy = (p.y - u.y) / 10.0
            d = math.hypot(p.x - u.x, p.y - u.y)
            chain_slack = max(0.0, cfg.pat_chain_max_distance - d) / cfg.pat_chain_max_distance
        v += [chain_partner_active, chain_pdx, chain_pdy, chain_slack]

        # Cross (3): 가장 가까운 안전 사분면 중심 상대 위치
        cross_active = 0.0
        safe_dx, safe_dy = 0.0, 0.0
        cx_m = cfg.map_width / 2.0; cy_m = cfg.map_height / 2.0
        quad_centers = {
            0: (cx_m * 0.5,        cy_m * 0.5),
            1: (cx_m * 1.5,        cy_m * 0.5),
            2: (cx_m * 0.5,        cy_m * 1.5),
            3: (cx_m * 1.5,        cy_m * 1.5),
        }
        for tg in self.boss.telegraphs:
            if tg.pattern_id == PatternID.CROSS_INFERNO:
                cross_active = 1.0
                best_d = 1e9
                for q in tg.target_unit_ids:
                    if 0 <= q < 4:
                        qx, qy = quad_centers[q]
                        d = math.hypot(qx - u.x, qy - u.y)
                        if d < best_d:
                            best_d = d
                            safe_dx = (qx - u.x) / 10.0
                            safe_dy = (qy - u.y) / 10.0
                break
        v += [cross_active, safe_dx, safe_dy]

        # Stagger (2)
        v += [1.0 if self.boss.stagger_active else 0.0,
              self.boss.stagger_gauge / max(1.0, cfg.stagger_gauge)]

        # Seal (4)
        seal_active = 0.0
        my_spot_dx, my_spot_dy, seal_progress = 0.0, 0.0, 0.0
        for tg in self.boss.telegraphs:
            if tg.pattern_id == PatternID.SEAL_BREAK:
                seal_active = 1.0
                spots = tg.extra.get("npc_spots", {})
                if uid in spots:
                    si = spots[uid]
                    if 0 <= si < len(tg.shapes):
                        s = tg.shapes[si]
                        my_spot_dx = (s.params["cx"] - u.x) / 10.0
                        my_spot_dy = (s.params["cy"] - u.y) / 10.0
                hold_prog = self._seal_hold_count / max(1, cfg.pat_seal_hold_turns)
                wind_prog = 1.0 - tg.turns_remaining / max(1, tg.total_wind_up)
                seal_progress = max(hold_prog, wind_prog)
                break
        v += [seal_active, my_spot_dx, my_spot_dy, seal_progress]

        # ── [8] Player (4) ──
        player = self.units[cfg.player_slot]
        pdx = player.x - u.x; pdy = player.y - u.y
        pdist = math.hypot(pdx, pdy)
        v += [pdx / 10.0, pdy / 10.0, pdist / 10.0,
              player.hp / max(1, player.max_hp)]

        arr = np.array(v, dtype=np.float32)
        expected = cfg.obs_size
        if arr.shape[0] < expected:
            arr = np.pad(arr, (0, expected - arr.shape[0]))
        elif arr.shape[0] > expected:
            arr = arr[:expected]
        return arr

    def _pattern_target_info(self, tg: ActiveTelegraph, pid: PatternID,
                             u: PartyUnit, uid: int) -> Tuple[float, float, float]:
        """패턴별 primary target의 (am_I_target, dx/10, dy/10) 반환."""
        am_I = 0.0
        tdx, tdy = 0.0, 0.0

        if pid in (PatternID.SLASH, PatternID.CHARGE,
                   PatternID.MARK, PatternID.TAIL_SWIPE):
            if tg.target_unit_ids:
                tuid = tg.target_unit_ids[0]
                am_I = 1.0 if tuid == uid else 0.0
                if tuid in self.units:
                    t = self.units[tuid]
                    tdx = (t.x - u.x) / 10.0
                    tdy = (t.y - u.y) / 10.0
        elif pid == PatternID.CURSED_CHAIN:
            if len(tg.target_unit_ids) >= 2:
                a, b = tg.target_unit_ids[:2]
                if uid == a or uid == b:
                    am_I = 1.0
                    partner = b if uid == a else a
                else:
                    partner = a
                if partner in self.units:
                    p = self.units[partner]
                    tdx = (p.x - u.x) / 10.0
                    tdy = (p.y - u.y) / 10.0
        elif pid == PatternID.ERUPTION:
            if tg.shapes:
                best = min(tg.shapes,
                           key=lambda s: math.hypot(s.params["cx"] - u.x,
                                                    s.params["cy"] - u.y))
                tdx = (best.params["cx"] - u.x) / 10.0
                tdy = (best.params["cy"] - u.y) / 10.0
        elif pid == PatternID.STAGGER:
            tdx = (self.boss.x - u.x) / 10.0
            tdy = (self.boss.y - u.y) / 10.0
        elif pid == PatternID.CROSS_INFERNO:
            cx_m = self.config.map_width / 2.0
            cy_m = self.config.map_height / 2.0
            quad_centers = {
                0: (cx_m * 0.5, cy_m * 0.5), 1: (cx_m * 1.5, cy_m * 0.5),
                2: (cx_m * 0.5, cy_m * 1.5), 3: (cx_m * 1.5, cy_m * 1.5),
            }
            best_d = 1e9
            for q in tg.target_unit_ids:
                if 0 <= q < 4:
                    qx, qy = quad_centers[q]
                    d = math.hypot(qx - u.x, qy - u.y)
                    if d < best_d:
                        best_d = d
                        tdx = (qx - u.x) / 10.0
                        tdy = (qy - u.y) / 10.0
        elif pid == PatternID.SEAL_BREAK:
            spots = tg.extra.get("npc_spots", {})
            if uid in spots and uid != self.config.player_slot:
                am_I = 1.0
                si = spots[uid]
                if 0 <= si < len(tg.shapes):
                    s = tg.shapes[si]
                    tdx = (s.params["cx"] - u.x) / 10.0
                    tdy = (s.params["cy"] - u.y) / 10.0

        return am_I, tdx, tdy

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
        return {
            "step": self.current_step,
            "boss": {
                "x": float(self.boss.x), "y": float(self.boss.y),
                "hp": int(self.boss.hp), "max_hp": int(self.config.boss_max_hp),
                "phase": int(self.boss.phase),
                "invuln": int(self.boss.invuln_turns), "grog": int(self.boss.grog_turns),
                "stagger_active": bool(self.boss.stagger_active),
                "stagger_gauge": float(self.boss.stagger_gauge),
                "radius": float(self.config.boss_radius),
                "vx": float(self.boss.x - self._prev_boss_pos[0]),
                "vy": float(self.boss.y - self._prev_boss_pos[1]),
            },
            "units": [
                {
                    "uid": int(u.uid), "role": int(u.role),
                    "x": float(u.x), "y": float(u.y),
                    "hp": int(u.hp), "max_hp": int(u.max_hp),
                    "alive": bool(u.alive),
                    "marked": bool(u.marked_turns > 0),
                    "chained_with": int(u.chained_with) if u.chained_with is not None else -1,
                    "buff_atk": int(u.buff_atk), "buff_shield": int(u.buff_shield),
                    "radius": float(u.radius),
                    "vx": float(u.x - self._prev_unit_positions.get(u.uid, (u.x, u.y))[0]),
                    "vy": float(u.y - self._prev_unit_positions.get(u.uid, (u.x, u.y))[1]),
                }
                for u in self.units.values()
            ],
            "telegraphs": [
                {
                    "pattern": int(tg.pattern_id),
                    "turns_remaining": int(tg.turns_remaining),
                    "total_wind_up": int(tg.total_wind_up),
                    "shapes": [s.to_dict() for s in tg.shapes],
                    "target_uids": [int(x) for x in tg.target_unit_ids],
                }
                for tg in self.boss.telegraphs
            ],
            "events": [
                {"uid": uid, **event}
                for uid, event_list in self.step_events.items()
                for event in event_list
            ],
            "done": bool(self.done),
            "victory": bool(self.victory),
            "wipe": bool(self.wipe),
        }
