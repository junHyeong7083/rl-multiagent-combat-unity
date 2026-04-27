"""보스 패턴 8종 — 유클리드 연속 공간 기하 판정

각 패턴은 `contains(pos)` 로 "해당 위치가 위험 영역 안에 있는가"를 판정한다.
Unity는 기하 파라미터(중심/반경/각도/선분)를 받아 쉐이더/Quad로 렌더링.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random

from .config import PatternID, BossConfig


Pos = Tuple[float, float]


# ─────────────────── 기하 유틸 ───────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _dist(a: Pos, b: Pos) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_in_circle(p: Pos, center: Pos, radius: float) -> bool:
    return _dist(p, center) <= radius


def _point_in_fan(p: Pos, origin: Pos, forward_rad: float,
                  angle_rad: float, radius: float) -> bool:
    dx = p[0] - origin[0]; dy = p[1] - origin[1]
    d2 = dx * dx + dy * dy
    if d2 > radius * radius or d2 < 1e-8:
        return d2 < 1e-8  # 원점 자체는 포함
    ang = math.atan2(dy, dx)
    diff = abs(ang - forward_rad)
    if diff > math.pi: diff = 2 * math.pi - diff
    return diff <= angle_rad * 0.5


def _point_in_line_segment(p: Pos, a: Pos, b: Pos, half_width: float) -> bool:
    """선분 a-b에서 half_width 이내 거리에 p가 있는지."""
    ax, ay = a; bx, by = b
    dx = bx - ax; dy = by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-8:
        return _dist(p, a) <= half_width
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / seg_len2
    t = _clamp(t, 0.0, 1.0)
    px = ax + t * dx; py = ay + t * dy
    return math.hypot(p[0] - px, p[1] - py) <= half_width


# ─────────────────── 도형 데이터 (Unity 송신용) ───────────────────

@dataclass
class Shape:
    """위험 영역 기하 — Unity에서 렌더링"""
    kind: str                             # "circle" | "fan" | "line" | "cross"
    params: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.kind, **self.params}


# ─────────────────── ActiveTelegraph ───────────────────

@dataclass
class ActiveTelegraph:
    pattern_id: PatternID
    turns_remaining: int
    total_wind_up: int
    shapes: List[Shape] = field(default_factory=list)          # 위험 영역 리스트
    target_unit_ids: List[int] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)
    impacted: bool = False
    post_impact_turns: int = 0

    def contains(self, pos: Pos) -> bool:
        """이 텔레그래프의 위험 영역에 pos가 들어가는지 (impact 판정용)."""
        for s in self.shapes:
            if _shape_contains(s, pos):
                return True
        return False


def _shape_contains(s: Shape, p: Pos) -> bool:
    if s.kind == "circle":
        return _point_in_circle(p, (s.params["cx"], s.params["cy"]), s.params["r"])
    if s.kind == "fan":
        return _point_in_fan(p, (s.params["cx"], s.params["cy"]),
                             s.params["angle"], s.params["width"], s.params["r"])
    if s.kind == "line":
        return _point_in_line_segment(p,
            (s.params["ax"], s.params["ay"]),
            (s.params["bx"], s.params["by"]),
            s.params["hw"])
    if s.kind == "cross":
        # 맵 중앙 십자. 안전 사분면 bitmask (bit 0=좌상, 1=우상, 2=좌하, 3=우하)
        cx = s.params["cx"]; cy = s.params["cy"]
        hw = s.params["hw"]
        safe_mask = int(s.params.get("safe_mask", 0))
        # 십자 영역 안?
        in_h = abs(p[1] - cy) <= hw
        in_v = abs(p[0] - cx) <= hw
        if not (in_h or in_v):
            # 사분면 안전 체크
            q = 0
            if p[0] >= cx: q |= 1
            if p[1] >= cy: q |= 2
            return (safe_mask >> q) & 1 == 0
        return True
    return False


# ─────────────────── 패턴 정의 ───────────────────

@dataclass
class BasePattern:
    pattern_id: PatternID
    name: str
    wind_up_turns: int
    cooldown: int
    base_damage: int

    def build_shapes(self, boss_pos: Pos, party: Dict[int, Pos],
                     cfg: BossConfig, rng: random.Random, extra: Dict
                     ) -> Tuple[List[Shape], List[int]]:
        raise NotImplementedError


class SlashPattern(BasePattern):
    def __init__(self):
        # wind_up 3: 근접이라 짧은 편 유지. 피해 50 → 35 (학습 진입 난이도↓)
        super().__init__(PatternID.SLASH, "Slash", 3, 3, 35)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        target_uid = extra.get("aggro_top_uid", 0)
        tpos = party.get(target_uid, (boss_pos[0] + 1, boss_pos[1]))
        angle = math.atan2(tpos[1] - boss_pos[1], tpos[0] - boss_pos[0])
        shape = Shape("fan", {
            "cx": boss_pos[0], "cy": boss_pos[1],
            "angle": angle,
            "width": math.radians(cfg.pat_slash_angle_deg),
            "r": cfg.pat_slash_range,
        })
        return [shape], [target_uid]


class ChargePattern(BasePattern):
    def __init__(self):
        # wind_up 5: 장거리 직선 회피 시간 확보. 피해 80 → 55
        super().__init__(PatternID.CHARGE, "Charge", 5, 6, 55)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        if not party:
            return [], []
        target_uid = rng.choice(list(party.keys()))
        tpos = party[target_uid]
        # 선분 = 보스 → 타겟. 조금 더 뒤까지 연장.
        dx = tpos[0] - boss_pos[0]; dy = tpos[1] - boss_pos[1]
        length = math.hypot(dx, dy)
        if length < 1e-4:
            return [], [target_uid]
        ex = tpos[0] + dx / length * 3.0
        ey = tpos[1] + dy / length * 3.0
        shape = Shape("line", {
            "ax": boss_pos[0], "ay": boss_pos[1],
            "bx": ex, "by": ey,
            "hw": cfg.pat_charge_width,
        })
        return [shape], [target_uid]


class EruptionPattern(BasePattern):
    def __init__(self):
        # wind_up 5. 피해 60 → 40 (여러 장판이라 누적 위험 있음)
        super().__init__(PatternID.ERUPTION, "Eruption", 5, 5, 40)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        centers: List[Pos] = []
        uids = list(party.keys()); rng.shuffle(uids)
        for uid in uids[:cfg.pat_eruption_count]:
            px, py = party[uid]
            cx = _clamp(px + rng.uniform(-2.0, 2.0), 1.5, cfg.map_width - 1.5)
            cy = _clamp(py + rng.uniform(-2.0, 2.0), 1.5, cfg.map_height - 1.5)
            centers.append((cx, cy))
        while len(centers) < cfg.pat_eruption_count:
            centers.append((rng.uniform(2.0, cfg.map_width - 2.0),
                           rng.uniform(2.0, cfg.map_height - 2.0)))
        shapes = [Shape("circle", {"cx": c[0], "cy": c[1], "r": cfg.pat_eruption_radius})
                  for c in centers]
        return shapes, []


class TailSwipePattern(BasePattern):
    def __init__(self):
        # wind_up 3: 보스 후방이라 딜러가 측면 이동 시간 확보
        super().__init__(PatternID.TAIL_SWIPE, "TailSwipe", 3, 5, 70)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        target_uid = extra.get("aggro_top_uid", 0)
        tpos = party.get(target_uid, (boss_pos[0] + 1, boss_pos[1]))
        # 어그로 방향의 반대편이 꼬리
        forward = math.atan2(tpos[1] - boss_pos[1], tpos[0] - boss_pos[0]) + math.pi
        shape = Shape("fan", {
            "cx": boss_pos[0], "cy": boss_pos[1],
            "angle": forward,
            "width": math.radians(cfg.pat_tail_angle_deg),
            "r": cfg.pat_tail_range,
        })
        return [shape], []


class MarkPattern(BasePattern):
    def __init__(self):
        # wind_up 6: 협동 기믹. 표식자는 멀리 도망, 나머지는 반대 방향 — 충분한 시간 필요
        super().__init__(PatternID.MARK, "Mark", 6, 8, 80)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        # 탱커 제외 랜덤 1명
        non_tanks = [uid for uid in party
                     if extra.get("roles", {}).get(uid) != 1]
        if not non_tanks:
            non_tanks = list(party.keys())
        target_uid = rng.choice(non_tanks) if non_tanks else 0
        # 폭발은 impact 시점에 대상 위치 기준 재계산. 사전 shapes는 빈 리스트.
        return [], [target_uid]


class StaggerPattern(BasePattern):
    def __init__(self):
        # wind_up 6: 파티 집결 + 300 게이지 깎기 시간 필요
        super().__init__(PatternID.STAGGER, "Stagger", 6, 12, 120)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        # 실패 시 전체. 사전 위험 없음.
        return [], []


class CrossInfernoPattern(BasePattern):
    def __init__(self):
        # wind_up 7: 전멸기. 안전 사분면까지 이동 시간 충분히
        super().__init__(PatternID.CROSS_INFERNO, "CrossInferno", 7, 15, 300)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        cx = cfg.map_width / 2.0
        cy = cfg.map_height / 2.0
        # 4 사분면 중 2곳 안전 (bit 0=좌상, 1=우상, 2=좌하, 3=우하)
        safe = rng.sample([0, 1, 2, 3], 2)
        safe_mask = 0
        for q in safe:
            safe_mask |= (1 << q)
        shape = Shape("cross", {
            "cx": cx, "cy": cy,
            "hw": cfg.pat_cross_band_half_width,
            "safe_mask": float(safe_mask),
        })
        return [shape], safe


class CursedChainPattern(BasePattern):
    def __init__(self):
        # wind_up 3: 연결 쌍이 서로 접근할 시간
        super().__init__(PatternID.CURSED_CHAIN, "CursedChain", 3, 10, 25)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        uids = list(party.keys())
        if len(uids) < 2:
            return [], []
        pair = rng.sample(uids, 2)
        return [], pair


class SealBreakPattern(BasePattern):
    """봉인 해제: 보스 주변 4장판 중 3개를 NPC가 각각 점유.
    딜러에 가장 가까운 장판은 "딜러용"으로 NPC 무시.
    """
    def __init__(self):
        # wind_up = arrive(20) + hold(15) = 35턴. impact은 매 턴 체크.
        super().__init__(PatternID.SEAL_BREAK, "SealBreak", 35, 999, 9999)
        # cooldown 999 = 페이즈 전이 시에만 수동 발동 (select_pattern으로 안 나옴)

    def build_shapes(self, boss_pos, party, cfg, rng, extra):
        # 보스 주변 동서남북 4개 장판
        d = cfg.pat_seal_spot_distance
        spot_offsets = [(d, 0), (-d, 0), (0, d), (0, -d)]
        spots = []
        for dx, dy in spot_offsets:
            sx = _clamp(boss_pos[0] + dx, 1.5, cfg.map_width - 1.5)
            sy = _clamp(boss_pos[1] + dy, 1.5, cfg.map_height - 1.5)
            spots.append((sx, sy))

        shapes = [Shape("circle", {"cx": s[0], "cy": s[1], "r": cfg.pat_seal_spot_radius})
                  for s in spots]

        # 딜러에 가장 가까운 장판 인덱스 저장 (NPC가 무시할 것)
        dealer_uid = extra.get("dealer_uid", 0)
        dealer_pos = party.get(dealer_uid, boss_pos)
        dealer_spot_idx = min(range(len(spots)),
                              key=lambda i: _dist(dealer_pos, spots[i]))

        return shapes, [dealer_spot_idx]  # target_unit_ids에 딜러 장판 인덱스 저장


PATTERN_REGISTRY: Dict[PatternID, BasePattern] = {
    PatternID.SLASH: SlashPattern(),
    PatternID.CHARGE: ChargePattern(),
    PatternID.ERUPTION: EruptionPattern(),
    PatternID.TAIL_SWIPE: TailSwipePattern(),
    PatternID.MARK: MarkPattern(),
    PatternID.STAGGER: StaggerPattern(),
    PatternID.CROSS_INFERNO: CrossInfernoPattern(),
    PatternID.CURSED_CHAIN: CursedChainPattern(),
    PatternID.SEAL_BREAK: SealBreakPattern(),
}


# ─────────────────── 8방향 위험 센서 (관찰용) ───────────────────

def sample_danger_sensor(pos: Pos,
                         telegraphs: List[ActiveTelegraph],
                         max_distance: float = 6.0,
                         step: float = 0.4) -> List[float]:
    """8방향으로 한 스텝씩 나아가며 가장 가까운 위험 지점까지 거리를 측정.

    반환: 8개 값 (N, NE, E, SE, S, SW, W, NW), 각각 [0, 1] (1=안전).
    """
    result = []
    for i in range(8):
        theta = i * math.pi / 4
        dx = math.cos(theta); dy = math.sin(theta)
        found = max_distance
        # 샘플링
        t = step
        while t <= max_distance:
            probe = (pos[0] + dx * t, pos[1] + dy * t)
            hit = False
            for tg in telegraphs:
                if tg.contains(probe):
                    hit = True
                    break
            if hit:
                found = t
                break
            t += step
        # 정규화: 멀수록 1 (안전), 가까울수록 0 (위험)
        result.append(found / max_distance)
    return result
