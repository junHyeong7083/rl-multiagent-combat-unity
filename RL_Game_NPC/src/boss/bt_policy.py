"""Layer 1 — 기믹 인식 Behavior Tree 정책 (v3 하이브리드용)

규칙이 명확한 협동 기믹 패턴에 대해 결정론적 대응 행동을 반환한다.
해당하는 규칙이 없으면 None 을 반환해 상위 dispatcher 가 Layer 2 RL 로 fall-through 하게 한다.

우선순위:
  1. SEAL_BREAK 활성   → 배정된 spot 으로 이동
  2. MARK 활성         → 표식자면 아군에서 멀어짐 / 아니면 표식자에게서 멀어짐
  3. CURSED_CHAIN 활성 → 파트너에게 접근
  4. CROSS_INFERNO     → 가장 가까운 안전 사분면 중심으로 이동
  5. STAGGER 창 열림   → 보스 근접 + 스킬 딜 all-in
  6. 임박 위험 (텔레그래프 2턴 이내 + 내가 영역 안) → 탈출 방향
  7. 그 외              → None (RL 에게 위임)
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

from .config import BossActionID, PartyRole, PatternID
from .patterns import ActiveTelegraph


def _move_toward(u_x: float, u_y: float, tx: float, ty: float) -> int:
    """현재 위치에서 (tx, ty) 방향으로 8방향 이동 액션 반환."""
    dx = tx - u_x
    dy = ty - u_y
    if abs(dx) < 0.2 and abs(dy) < 0.2:
        return int(BossActionID.STAY)
    # 대각/직선 판정: 한 축이 다른 축의 2배 이상이면 직선, 아니면 대각
    ax, ay = abs(dx), abs(dy)
    if ax > ay * 2:
        return int(BossActionID.MOVE_RIGHT) if dx > 0 else int(BossActionID.MOVE_LEFT)
    if ay > ax * 2:
        return int(BossActionID.MOVE_DOWN) if dy > 0 else int(BossActionID.MOVE_UP)
    if dx > 0 and dy > 0:
        return int(BossActionID.MOVE_DOWN_RIGHT)
    if dx > 0 and dy < 0:
        return int(BossActionID.MOVE_UP_RIGHT)
    if dx < 0 and dy > 0:
        return int(BossActionID.MOVE_DOWN_LEFT)
    return int(BossActionID.MOVE_UP_LEFT)


def _move_away(u_x: float, u_y: float, from_x: float, from_y: float,
               distance: float = 5.0) -> int:
    dx = u_x - from_x
    dy = u_y - from_y
    d = math.hypot(dx, dy)
    if d < 0.01:
        return int(BossActionID.MOVE_LEFT)
    target_x = u_x + dx / d * distance
    target_y = u_y + dy / d * distance
    return _move_toward(u_x, u_y, target_x, target_y)


class BTPolicy:
    """Layer 1 BT — 기믹 인식 정책.

    사용법:
        bt = BTPolicy(env, uid)
        act = bt.act()
        if act is None:
            act = rl_policy.act()   # Layer 2 RL fall-through
    """

    def __init__(self, env, uid: int):
        self.env = env
        self.uid = uid

    # ────────────── 진입점 ──────────────

    def act(self) -> Optional[int]:
        env = self.env
        u = env.units[self.uid]
        if not u.alive:
            return int(BossActionID.STAY)

        # Priority 0: user_test_mode 전투 미시작 시 → 플레이어 따라가기
        if env.config.user_test_mode and not env.combat_started:
            return self._follow_player(u)

        # Priority 1: SEAL_BREAK (페이즈 전이, 최우선)
        for tg in env.boss.telegraphs:
            if tg.pattern_id == PatternID.SEAL_BREAK:
                return self._handle_seal(tg, u)

        # Priority 2: STAGGER — 열리면 전원 딜 집중 (MMORPG 관례, 다른 기믹 무시)
        if env.boss.stagger_active:
            a = self._handle_stagger(u)
            if a is not None:
                return a

        # Priority 3: MARK
        for tg in env.boss.telegraphs:
            if tg.pattern_id == PatternID.MARK and tg.turns_remaining > 0:
                a = self._handle_mark(tg, u)
                if a is not None:
                    return a

        # Priority 4: CURSED_CHAIN
        for tg in env.boss.telegraphs:
            if tg.pattern_id == PatternID.CURSED_CHAIN and tg.turns_remaining > 0:
                a = self._handle_chain(tg, u)
                if a is not None:
                    return a

        # Priority 5: CROSS_INFERNO (wind_up 7턴 중 임박 구간만 BT)
        for tg in env.boss.telegraphs:
            if tg.pattern_id == PatternID.CROSS_INFERNO and tg.turns_remaining <= 4:
                a = self._handle_cross(tg, u)
                if a is not None:
                    return a

        # Priority 6: 임박 위험 (영역 안 + 2턴 이내)
        a = self._handle_immediate_danger(u)
        if a is not None:
            return a

        # Priority 7: 탱커 어그로 관리 — 어그로 잃으면 TAUNT
        if u.role == PartyRole.TANK:
            top_uid = env.boss.top_aggro_uid()
            if top_uid != self.uid and u.cooldowns.get(int(BossActionID.TAUNT), 0) <= 0:
                return int(BossActionID.TAUNT)

        # Fall-through → Layer 2 RL
        return None

    # ────────────── 패턴별 핸들러 ──────────────

    def _follow_player(self, u) -> int:
        """user_test_mode 전투 미시작: 플레이어 일정 거리 유지하며 따라감."""
        player = self.env.units[self.env.config.player_slot]
        d = math.hypot(u.x - player.x, u.y - player.y)
        # 너무 멀면 따라가고, 가까우면 STAY
        if d > 2.5:
            return _move_toward(u.x, u.y, player.x, player.y)
        return int(BossActionID.STAY)

    def _handle_seal(self, tg: ActiveTelegraph, u) -> int:
        cfg = self.env.config
        spots = tg.extra.get("npc_spots", {})
        # 딜러는 dealer_spot_idx (target_unit_ids[0]), NPC 는 spots[uid]
        if self.uid == cfg.player_slot:
            si = tg.target_unit_ids[0] if tg.target_unit_ids else -1
        else:
            si = spots.get(self.uid, -1)

        if not (0 <= si < len(tg.shapes)):
            return int(BossActionID.STAY)

        s = tg.shapes[si]
        sx, sy = s.params["cx"], s.params["cy"]

        # 내 spot 위?
        if math.hypot(u.x - sx, u.y - sy) <= s.params["r"]:
            # 보스 사거리 안이면 딜, 아니면 자리 사수
            if self.env._boss_dist(u.x, u.y) <= u.attack_range:
                if u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) <= 0:
                    return int(BossActionID.ATTACK_SKILL)
                return int(BossActionID.ATTACK_BASIC)
            return int(BossActionID.STAY)

        # 아직 도달 못 함 → 이동
        return _move_toward(u.x, u.y, sx, sy)

    def _handle_mark(self, tg: ActiveTelegraph, u) -> Optional[int]:
        if not tg.target_unit_ids:
            return None
        cfg = self.env.config
        mark_uid = tg.target_unit_ids[0]

        if mark_uid == self.uid:
            # 내가 표식자 → 나머지 아군 중심에서 멀어짐
            others = [x for x in self.env.units.values()
                      if x.uid != self.uid and x.alive]
            if not others:
                return None
            cx = sum(x.x for x in others) / len(others)
            cy = sum(x.y for x in others) / len(others)
            # 이미 충분히 멀면 fall-through
            min_d = min(math.hypot(u.x - x.x, u.y - x.y) for x in others)
            if min_d >= cfg.pat_mark_escape_distance:
                return None
            return _move_away(u.x, u.y, cx, cy, distance=5.0)

        # 내가 표식자 아님 → 표식자에게서 멀어짐
        if mark_uid not in self.env.units:
            return None
        mu = self.env.units[mark_uid]
        d = math.hypot(u.x - mu.x, u.y - mu.y)
        if d >= cfg.pat_mark_escape_distance:
            return None  # 이미 안전, RL 에게 위임
        return _move_away(u.x, u.y, mu.x, mu.y, distance=5.0)

    def _handle_chain(self, tg: ActiveTelegraph, u) -> Optional[int]:
        if len(tg.target_unit_ids) < 2:
            return None
        a_uid, b_uid = tg.target_unit_ids[:2]
        if self.uid not in (a_uid, b_uid):
            return None
        partner = b_uid if self.uid == a_uid else a_uid
        if partner not in self.env.units:
            return None
        p = self.env.units[partner]
        if not p.alive:
            return None
        d = math.hypot(u.x - p.x, u.y - p.y)
        # 여유 있으면 RL 에게 위임 (전투 우선)
        if d <= self.env.config.pat_chain_max_distance * 0.7:
            return None
        return _move_toward(u.x, u.y, p.x, p.y)

    def _handle_cross(self, tg: ActiveTelegraph, u) -> Optional[int]:
        cfg = self.env.config
        safe_quads = tg.target_unit_ids
        cx_m = cfg.map_width / 2.0
        cy_m = cfg.map_height / 2.0
        quad_centers = {
            0: (cx_m * 0.5, cy_m * 0.5),
            1: (cx_m * 1.5, cy_m * 0.5),
            2: (cx_m * 0.5, cy_m * 1.5),
            3: (cx_m * 1.5, cy_m * 1.5),
        }
        my_q = self.env._quadrant(u.x, u.y)

        # 십자 밴드 안이면 무조건 탈출
        in_band = (abs(u.x - cx_m) <= cfg.pat_cross_band_half_width
                   or abs(u.y - cy_m) <= cfg.pat_cross_band_half_width)
        if my_q in safe_quads and not in_band:
            return None  # 이미 안전, RL 에게 위임

        # 가장 가까운 안전 사분면 중심으로 이동
        best_d = 1e9
        tx, ty = u.x, u.y
        for q in safe_quads:
            if q in quad_centers:
                qx, qy = quad_centers[q]
                d = math.hypot(qx - u.x, qy - u.y)
                if d < best_d:
                    best_d = d
                    tx, ty = qx, qy
        return _move_toward(u.x, u.y, tx, ty)

    def _handle_stagger(self, u) -> Optional[int]:
        """스태거 창 열림 → 보스 근접 + 공격 우선."""
        env = self.env
        boss_dist = env._boss_dist(u.x, u.y)

        # 딜 사거리 밖이면 접근
        if boss_dist > u.attack_range:
            return _move_toward(u.x, u.y, env.boss.x, env.boss.y)

        # 사거리 내 — 탱커는 taunt 도 스태거 기여에 포함되므로 쿨다운 체크
        if u.role == PartyRole.TANK:
            if u.cooldowns.get(int(BossActionID.TAUNT), 0) <= 0:
                return int(BossActionID.TAUNT)
        # 스킬 우선, 없으면 기본
        if u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) <= 0:
            return int(BossActionID.ATTACK_SKILL)
        return int(BossActionID.ATTACK_BASIC)

    def _handle_immediate_danger(self, u) -> Optional[int]:
        """임박 위험(2턴 이내 impact + 내가 영역 안) 시 탈출 방향."""
        env = self.env
        danger_tgs = [tg for tg in env.boss.telegraphs
                      if tg.pattern_id != PatternID.SEAL_BREAK
                      and tg.turns_remaining <= 2
                      and tg.contains((u.x, u.y))]
        if not danger_tgs:
            return None
        urgent_tg = min(danger_tgs, key=lambda t: t.turns_remaining)

        # 8방향 중 가장 가까이 빠져나갈 수 있는 방향 찾기
        best_step = 1e9
        best_dir: Optional[Tuple[float, float]] = None
        for di in range(8):
            theta = di * math.pi / 4
            for step_m in (0.5, 1.0, 1.5, 2.0, 3.0):
                tx = u.x + math.cos(theta) * step_m
                ty = u.y + math.sin(theta) * step_m
                if not urgent_tg.contains((tx, ty)):
                    if step_m < best_step:
                        best_step = step_m
                        best_dir = (math.cos(theta), math.sin(theta))
                    break

        if best_dir is None:
            return int(BossActionID.STAY)
        dx, dy = best_dir
        target_x = u.x + dx * 3.0
        target_y = u.y + dy * 3.0
        return _move_toward(u.x, u.y, target_x, target_y)
