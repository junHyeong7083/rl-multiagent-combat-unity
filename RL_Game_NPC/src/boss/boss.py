"""보스 클래스 — FSM, 어그로, 스태거, 페이즈 관리"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random

from .config import BossConfig, PatternID, PhaseID
from .patterns import ActiveTelegraph, BasePattern, PATTERN_REGISTRY, Pos


@dataclass
class Boss:
    config: BossConfig
    x: int = 10
    y: int = 10
    hp: int = 0
    phase: PhaseID = PhaseID.P1
    invuln_turns: int = 0                # 페이즈 전이 직후 무적
    grog_turns: int = 0                  # 스태거 성공 후 그로기
    stagger_active: bool = False
    stagger_gauge: float = 0.0

    # 어그로 테이블: unit_id -> 수치
    aggro: Dict[int, float] = field(default_factory=dict)

    # 활성 텔레그래프
    telegraphs: List[ActiveTelegraph] = field(default_factory=list)

    # 패턴 쿨타임 (pattern_id → 남은 턴)
    cooldowns: Dict[int, int] = field(default_factory=dict)

    # 최근 사용 패턴 로그 (연속 사용 방지)
    recent_patterns: List[int] = field(default_factory=list)

    # RNG
    rng: random.Random = field(default_factory=random.Random)

    # 지속형 효과 — 저주 연결(활성 여부, 페어, 잔여 턴)
    active_chain: Optional[Dict] = None

    def __post_init__(self):
        if self.hp <= 0:
            self.hp = self.config.boss_max_hp
        self.x = self.config.map_width // 2
        self.y = self.config.map_height // 2

    # ─────────────────── 어그로 ───────────────────

    def add_aggro(self, uid: int, amount: float):
        self.aggro[uid] = self.aggro.get(uid, 0.0) + amount

    def decay_aggro(self):
        for k in list(self.aggro.keys()):
            self.aggro[k] *= self.config.aggro_decay

    def top_aggro_uid(self) -> Optional[int]:
        if not self.aggro:
            return None
        return max(self.aggro.items(), key=lambda kv: kv[1])[0]

    # ─────────────────── 페이즈 전이 ───────────────────

    def check_phase_transition(self) -> bool:
        """HP 임계값 도달 시 다음 페이즈로. 전이 시 True 반환."""
        hp_ratio = self.hp / self.config.boss_max_hp
        th = self.config.phase_hp_thresholds
        new_phase = self.phase
        if self.phase == PhaseID.P1 and hp_ratio <= th[0]:
            new_phase = PhaseID.P2
        elif self.phase == PhaseID.P2 and hp_ratio <= th[1]:
            new_phase = PhaseID.P3
        if new_phase != self.phase:
            self.phase = new_phase
            self.invuln_turns = self.config.phase_transition_invuln_turns
            # 중앙 귀환
            self.x = self.config.map_width // 2
            self.y = self.config.map_height // 2
            # 모든 텔레그래프 해제
            self.telegraphs.clear()
            return True
        return False

    # ─────────────────── 쿨타임 ───────────────────

    def tick_cooldowns(self):
        for k in list(self.cooldowns.keys()):
            self.cooldowns[k] = max(0, self.cooldowns[k] - 1)

    def _can_use(self, pid: int) -> bool:
        return self.cooldowns.get(pid, 0) <= 0

    # ─────────────────── 패턴 선택 ───────────────────

    def select_pattern(self) -> Optional[PatternID]:
        weights = self.config.pattern_weights[int(self.phase)]
        candidates = []
        probs = []
        for pid_int, w in weights.items():
            if w <= 0: continue
            if not self._can_use(pid_int): continue
            # 동시 활성 2개 제한
            if len(self.telegraphs) >= self.config.max_active_telegraphs:
                return None
            # 같은 패턴 활성 중복 방지
            if any(t.pattern_id == PatternID(pid_int) for t in self.telegraphs):
                continue
            # 최근 3회 사용 패턴 가중치 절반
            penalty = 0.5 if pid_int in self.recent_patterns[-3:] else 1.0
            candidates.append(pid_int)
            probs.append(w * penalty)
        if not candidates:
            return None
        total = sum(probs)
        r = self.rng.random() * total
        acc = 0.0
        for pid_int, p in zip(candidates, probs):
            acc += p
            if r <= acc:
                return PatternID(pid_int)
        return PatternID(candidates[-1])

    # ─────────────────── 텔레그래프 시작 ───────────────────

    def start_pattern(
        self,
        pid: PatternID,
        party_positions: Dict[int, Pos],
        party_roles: Dict[int, int],
    ):
        pat: BasePattern = PATTERN_REGISTRY[pid]
        scale = self.config.phase_cooldown_scale[int(self.phase)]
        self.cooldowns[int(pid)] = int(pat.cooldown * scale)
        self.recent_patterns.append(int(pid))
        if len(self.recent_patterns) > 10:
            self.recent_patterns.pop(0)

        extra = {
            "aggro_top_uid": self.top_aggro_uid() or 0,
            "roles": party_roles,
        }
        tiles, targets = pat.select_tiles(
            (self.x, self.y),
            party_positions,
            self.config.map_width, self.config.map_height,
            self.rng, extra,
        )
        tg = ActiveTelegraph(
            pattern_id=pid,
            turns_remaining=pat.wind_up_turns,
            total_wind_up=pat.wind_up_turns,
            danger_tiles=tiles,
            target_unit_ids=list(targets),
            extra={"damage": pat.base_damage},
        )
        # 스태거는 시작 시 게이지 초기화
        if pid == PatternID.STAGGER:
            self.stagger_active = True
            self.stagger_gauge = self.config.stagger_gauge
        self.telegraphs.append(tg)

    # ─────────────────── 텔레그래프 틱 ───────────────────

    def tick_telegraphs(self) -> List[ActiveTelegraph]:
        """턴 시작 시 텔레그래프 카운트다운. 발동 직전(0턴)인 것들 반환."""
        ready = []
        for tg in self.telegraphs:
            tg.turns_remaining -= 1
            if tg.turns_remaining <= 0 and not tg.impacted:
                ready.append(tg)
        return ready

    def finalize_telegraphs(self):
        """impact 처리 완료된 텔레그래프 제거 (지속형은 유지)"""
        keep = []
        for tg in self.telegraphs:
            if tg.impacted and tg.post_impact_turns <= 0:
                continue
            if tg.post_impact_turns > 0:
                tg.post_impact_turns -= 1
                keep.append(tg)
            elif not tg.impacted:
                keep.append(tg)
        self.telegraphs = keep

    # ─────────────────── 이동 ───────────────────

    def move_toward_top_aggro(self, occupied: Set[Pos]):
        if self.invuln_turns > 0 or self.grog_turns > 0:
            return
        top = self.top_aggro_uid()
        if top is None:
            return
        target = occupied.get(top) if isinstance(occupied, dict) else None
        # occupied가 dict가 아니라 set으로 온 경우 상위에서 직접 target 전달 필요
        # 여기서는 단순 제자리 유지로 폴백

    def move_toward(self, tx: int, ty: int, blocked: Set[Pos]):
        if self.invuln_turns > 0 or self.grog_turns > 0:
            return
        dx = (1 if tx > self.x else (-1 if tx < self.x else 0))
        dy = (1 if ty > self.y else (-1 if ty < self.y else 0))
        # x축 우선
        nx, ny = self.x + dx, self.y
        if (nx, ny) not in blocked and 0 <= nx < self.config.map_width:
            self.x = nx; return
        nx, ny = self.x, self.y + dy
        if (nx, ny) not in blocked and 0 <= ny < self.config.map_height:
            self.y = ny

    # ─────────────────── 피격 ───────────────────

    def take_damage(self, amount: int, attacker_uid: int) -> int:
        if self.invuln_turns > 0:
            return 0
        dmg = max(1, amount - self.config.boss_defense)
        actual = min(dmg, self.hp)
        self.hp -= actual
        self.add_aggro(attacker_uid, actual * self.config.aggro_damage_weight)
        return actual

    def tick_end_of_turn(self):
        if self.invuln_turns > 0:
            self.invuln_turns -= 1
        if self.grog_turns > 0:
            self.grog_turns -= 1
        self.decay_aggro()
        self.tick_cooldowns()
