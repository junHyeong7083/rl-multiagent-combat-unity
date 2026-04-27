"""보스 클래스 — FSM, 어그로, 스태거, 페이즈 (유클리드 연속 공간)"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random

from .config import BossConfig, PatternID, PhaseID
from .patterns import ActiveTelegraph, BasePattern, PATTERN_REGISTRY, Pos, Shape


@dataclass
class Boss:
    config: BossConfig
    x: float = 10.0
    y: float = 10.0
    hp: int = 0
    phase: PhaseID = PhaseID.P1
    invuln_turns: int = 0
    grog_turns: int = 0
    stagger_active: bool = False
    stagger_gauge: float = 0.0

    aggro: Dict[int, float] = field(default_factory=dict)
    telegraphs: List[ActiveTelegraph] = field(default_factory=list)
    cooldowns: Dict[int, int] = field(default_factory=dict)
    recent_patterns: List[int] = field(default_factory=list)

    rng: random.Random = field(default_factory=random.Random)
    active_chain: Optional[Dict] = None

    def __post_init__(self):
        if self.hp <= 0:
            self.hp = self.config.boss_max_hp
        self.x = self.config.map_width / 2.0
        self.y = self.config.map_height / 2.0

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
            self.x = self.config.map_width / 2.0
            self.y = self.config.map_height / 2.0
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
            if len(self.telegraphs) >= self.config.max_active_telegraphs:
                return None
            if any(t.pattern_id == PatternID(pid_int) for t in self.telegraphs):
                continue
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

    def start_pattern(self, pid: PatternID,
                      party_positions: Dict[int, Pos],
                      party_roles: Dict[int, int],
                      extra_override: Optional[Dict] = None):
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
        if extra_override:
            extra.update(extra_override)
        shapes, targets = pat.build_shapes(
            (self.x, self.y),
            party_positions,
            self.config,
            self.rng, extra,
        )
        tg = ActiveTelegraph(
            pattern_id=pid,
            turns_remaining=pat.wind_up_turns,
            total_wind_up=pat.wind_up_turns,
            shapes=shapes,
            target_unit_ids=list(targets),
            extra={"damage": pat.base_damage},
        )
        if pid == PatternID.STAGGER:
            self.stagger_active = True
            self.stagger_gauge = self.config.stagger_gauge

        # SEAL_BREAK: NPC ↔ spot 최적 배정 (총 이동거리 최소 + 보스 관통 경로 패널티)
        if pid == PatternID.SEAL_BREAK:
            from itertools import permutations
            dealer_spot_idx = tg.target_unit_ids[0] if tg.target_unit_ids else -1
            available = [i for i in range(len(tg.shapes)) if i != dealer_spot_idx]
            npc_uids = sorted(uid for uid in party_positions
                              if uid != self.config.player_slot)

            def _path_blocked_by_boss(nx, ny, sx, sy):
                """NPC → spot 직선 경로가 보스 원을 통과하는지 체크."""
                dx = sx - nx; dy = sy - ny
                seg2 = dx * dx + dy * dy
                if seg2 < 1e-6:
                    return False
                t = max(0.0, min(1.0, ((self.x - nx) * dx + (self.y - ny) * dy) / seg2))
                px = nx + t * dx; py = ny + t * dy
                return math.hypot(self.x - px, self.y - py) < self.config.boss_radius + 0.3

            def _cost(npc_uid, si):
                px, py = party_positions[npc_uid]
                sx = tg.shapes[si].params["cx"]
                sy = tg.shapes[si].params["cy"]
                d = math.hypot(px - sx, py - sy)
                if _path_blocked_by_boss(px, py, sx, sy):
                    d += 10.0  # 보스 관통 경로 강한 패널티
                return d

            # 3 NPC × 3 spot 모든 순열 중 총 비용 최소
            best_perm, best_total = None, 1e18
            for perm in permutations(available, len(npc_uids)):
                total = sum(_cost(uid, si) for uid, si in zip(npc_uids, perm))
                if total < best_total:
                    best_total, best_perm = total, perm
            assignments: Dict[int, int] = dict(zip(npc_uids, best_perm)) if best_perm else {}
            tg.extra["npc_spots"] = assignments

        self.telegraphs.append(tg)

    # ─────────────────── 텔레그래프 틱 ───────────────────

    def tick_telegraphs(self) -> List[ActiveTelegraph]:
        ready = []
        for tg in self.telegraphs:
            tg.turns_remaining -= 1
            if tg.turns_remaining <= 0 and not tg.impacted:
                ready.append(tg)
        return ready

    def finalize_telegraphs(self):
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

    # ─────────────────── 이동 (유클리드) ───────────────────

    def move_toward(self, tx: float, ty: float,
                    occupied_positions: List[Tuple[float, float, float]]):
        """occupied_positions: [(x, y, radius), ...] 다른 유닛들.
        보스는 원형 충돌 체크로 겹치지 않게 이동."""
        if self.invuln_turns > 0 or self.grog_turns > 0:
            return
        if self.stagger_active:
            return  # 스태거 wind_up 중 보스 정지 (파티 집결·딜 타임 확보)
        dx = tx - self.x; dy = ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            return
        step = min(self.config.boss_move_speed, dist)
        nx = self.x + dx / dist * step
        ny = self.y + dy / dist * step
        # 맵 경계 클램프
        nx = max(self.config.boss_radius, min(self.config.map_width - self.config.boss_radius, nx))
        ny = max(self.config.boss_radius, min(self.config.map_height - self.config.boss_radius, ny))
        # 충돌 체크 (간단: 겹치면 이동 거리 축소해서 다시 시도)
        blocked = False
        for ox, oy, orad in occupied_positions:
            if math.hypot(nx - ox, ny - oy) < self.config.boss_radius + orad - 0.05:
                blocked = True; break
        if not blocked:
            self.x, self.y = nx, ny

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
