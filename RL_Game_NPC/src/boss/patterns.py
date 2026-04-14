"""보스 패턴 8종 구현

각 패턴은 3단계: WIND_UP(텔레그래프) → IMPACT(판정) → RECOVERY(회복).
격자 좌표 기반으로 위험 타일 집합과 대상을 계산한다.
Unity는 이 데이터(JSON)를 받아 시각 이펙트만 렌더링.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random

from .config import PatternID, BossConfig


# 격자 좌표 타입
Pos = Tuple[int, int]


@dataclass
class ActiveTelegraph:
    """현재 활성 중인 텔레그래프 인스턴스"""
    pattern_id: PatternID
    turns_remaining: int                 # 발동까지 남은 턴
    total_wind_up: int
    danger_tiles: Set[Pos] = field(default_factory=set)
    target_unit_ids: List[int] = field(default_factory=list)  # 지목 대상
    extra: Dict = field(default_factory=dict)   # 패턴별 추가 정보
    impacted: bool = False
    # 지속형 효과용 (연결, 십자 등)
    post_impact_turns: int = 0


@dataclass
class BasePattern:
    """패턴 기본 클래스 (데이터 + 로직)"""
    pattern_id: PatternID
    name: str
    wind_up_turns: int
    cooldown: int
    base_damage: int

    def select_tiles(
        self,
        boss_pos: Pos,
        party: Dict[int, Pos],
        map_w: int, map_h: int,
        rng: random.Random,
        extra: Dict,
    ) -> Tuple[Set[Pos], List[int]]:
        """이 패턴의 위험 타일과 대상 유닛 결정. 오버라이드 필수."""
        raise NotImplementedError


# ─────────────────────── 유틸 ───────────────────────

def _in_bounds(p: Pos, w: int, h: int) -> bool:
    return 0 <= p[0] < w and 0 <= p[1] < h


def _rect(cx: int, cy: int, rx: int, ry: int, w: int, h: int) -> Set[Pos]:
    out = set()
    for x in range(cx - rx, cx + rx + 1):
        for y in range(cy - ry, cy + ry + 1):
            if _in_bounds((x, y), w, h):
                out.add((x, y))
    return out


def _line(a: Pos, b: Pos, width: int, w: int, h: int) -> Set[Pos]:
    """a→b 선분 주변 width 너비 타일 (간단 Bresenham)"""
    x0, y0 = a; x1, y1 = b
    tiles = set()
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
    err = dx - dy
    cx, cy = x0, y0
    steps = 0
    while True:
        for ox in range(-width, width + 1):
            for oy in range(-width, width + 1):
                p = (cx + ox, cy + oy)
                if _in_bounds(p, w, h):
                    tiles.add(p)
        if (cx, cy) == (x1, y1) or steps > max(w, h) * 2:
            break
        e2 = 2 * err
        if e2 > -dy: err -= dy; cx += sx
        if e2 < dx:  err += dx; cy += sy
        steps += 1
    return tiles


# ─────────────────────── 패턴 구현 ───────────────────────

class SlashPattern(BasePattern):
    def __init__(self):
        super().__init__(PatternID.SLASH, "Slash", 2, 3, 50)

    def select_tiles(self, boss_pos, party, map_w, map_h, rng, extra):
        # 어그로 1위 방향 3x1 부채꼴 (간략화: 보스 전방 3칸)
        target_uid = extra.get("aggro_top_uid", 0)
        tpos = party.get(target_uid, boss_pos)
        dx = 1 if tpos[0] > boss_pos[0] else (-1 if tpos[0] < boss_pos[0] else 0)
        dy = 1 if tpos[1] > boss_pos[1] else (-1 if tpos[1] < boss_pos[1] else 0)
        if dx == 0 and dy == 0:
            dx = 1
        tiles = set()
        for step in range(1, 4):
            for ortho in (-1, 0, 1):
                if dx != 0:
                    p = (boss_pos[0] + dx * step, boss_pos[1] + ortho)
                else:
                    p = (boss_pos[0] + ortho, boss_pos[1] + dy * step)
                if _in_bounds(p, map_w, map_h):
                    tiles.add(p)
        return tiles, [target_uid]


class ChargePattern(BasePattern):
    def __init__(self):
        super().__init__(PatternID.CHARGE, "Charge", 3, 6, 80)

    def select_tiles(self, boss_pos, party, map_w, map_h, rng, extra):
        if not party:
            return set(), []
        target_uid = rng.choice(list(party.keys()))
        tpos = party[target_uid]
        return _line(boss_pos, tpos, 0, map_w, map_h), [target_uid]


class EruptionPattern(BasePattern):
    """3개 장판. 유닛 위치 기반으로 오프셋."""
    def __init__(self):
        super().__init__(PatternID.ERUPTION, "Eruption", 3, 5, 60)

    def select_tiles(self, boss_pos, party, map_w, map_h, rng, extra):
        centers = []
        uids = list(party.keys())
        rng.shuffle(uids)
        for uid in uids[:3]:
            px, py = party[uid]
            cx = max(1, min(map_w - 2, px + rng.randint(-2, 2)))
            cy = max(1, min(map_h - 2, py + rng.randint(-2, 2)))
            centers.append((cx, cy))
        # 부족하면 랜덤 좌표로 채움
        while len(centers) < 3:
            centers.append((rng.randint(2, map_w - 3), rng.randint(2, map_h - 3)))
        tiles = set()
        for c in centers:
            tiles |= _rect(c[0], c[1], 1, 1, map_w, map_h)
        return tiles, []


class TailSwipePattern(BasePattern):
    """보스 후방 5x3 (어그로 1위 반대 방향)"""
    def __init__(self):
        super().__init__(PatternID.TAIL_SWIPE, "TailSwipe", 2, 5, 70)

    def select_tiles(self, boss_pos, party, map_w, map_h, rng, extra):
        target_uid = extra.get("aggro_top_uid", 0)
        tpos = party.get(target_uid, (boss_pos[0] + 1, boss_pos[1]))
        dx = 1 if tpos[0] > boss_pos[0] else (-1 if tpos[0] < boss_pos[0] else 0)
        dy = 1 if tpos[1] > boss_pos[1] else (-1 if tpos[1] < boss_pos[1] else 0)
        # 후방 = 반대 방향
        bx, by = boss_pos
        tiles = set()
        if dx != 0:
            for step in range(1, 6):
                for ortho in (-1, 0, 1):
                    p = (bx - dx * step, by + ortho)
                    if _in_bounds(p, map_w, map_h):
                        tiles.add(p)
        else:
            for step in range(1, 6):
                for ortho in (-1, 0, 1):
                    p = (bx + ortho, by - dy * step)
                    if _in_bounds(p, map_w, map_h):
                        tiles.add(p)
        return tiles, []


class MarkPattern(BasePattern):
    """비탱커 1명에 4턴 표식. 5칸+ 이탈 못 하면 5x5 폭발."""
    def __init__(self):
        super().__init__(PatternID.MARK, "Mark", 4, 8, 80)

    def select_tiles(self, boss_pos, party, map_w, map_h, rng, extra):
        # 탱커 제외한 유닛 중 1명 선정 (첫 호출 시만)
        non_tanks = [uid for uid, _ in party.items() if extra.get("roles", {}).get(uid) != 1]
        if not non_tanks:
            non_tanks = list(party.keys())
        target_uid = rng.choice(non_tanks) if non_tanks else 0
        # 폭발 영역은 impact 시점 재계산하므로 여기서는 빈 tiles 반환
        return set(), [target_uid]


class StaggerPattern(BasePattern):
    """스태거 체크. 판정 타일은 광역 (파훼 실패 시)."""
    def __init__(self):
        super().__init__(PatternID.STAGGER, "Stagger", 4, 12, 120)

    def select_tiles(self, boss_pos, party, map_w, map_h, rng, extra):
        # 실패 시 전체 피해. 사전 위험 타일은 없음 (전체).
        return set(), []


class CrossInfernoPattern(BasePattern):
    """십자 전멸기. 맵 중앙 가로/세로 3줄 = 십자. 4사분면 중 2곳 안전."""
    def __init__(self):
        super().__init__(PatternID.CROSS_INFERNO, "CrossInferno", 5, 15, 300)

    def select_tiles(self, boss_pos, party, map_w, map_h, rng, extra):
        cx, cy = map_w // 2, map_h // 2
        tiles = set()
        # 가로 중앙 3줄
        for x in range(map_w):
            for y in (cy - 1, cy, cy + 1):
                if _in_bounds((x, y), map_w, map_h):
                    tiles.add((x, y))
        # 세로 중앙 3줄
        for y in range(map_h):
            for x in (cx - 1, cx, cx + 1):
                if _in_bounds((x, y), map_w, map_h):
                    tiles.add((x, y))
        # 안전 사분면 2곳 선정 (0:좌상, 1:우상, 2:좌하, 3:우하)
        safe = rng.sample([0, 1, 2, 3], 2)
        return tiles, safe


class CursedChainPattern(BasePattern):
    """랜덤 2명 연결. 6턴 지속. 거리 3칸 초과 시 매 턴 피해."""
    def __init__(self):
        super().__init__(PatternID.CURSED_CHAIN, "CursedChain", 2, 10, 25)

    def select_tiles(self, boss_pos, party, map_w, map_h, rng, extra):
        uids = list(party.keys())
        if len(uids) < 2:
            return set(), []
        pair = rng.sample(uids, 2)
        return set(), pair


PATTERN_REGISTRY: Dict[PatternID, BasePattern] = {
    PatternID.SLASH: SlashPattern(),
    PatternID.CHARGE: ChargePattern(),
    PatternID.ERUPTION: EruptionPattern(),
    PatternID.TAIL_SWIPE: TailSwipePattern(),
    PatternID.MARK: MarkPattern(),
    PatternID.STAGGER: StaggerPattern(),
    PatternID.CROSS_INFERNO: CrossInfernoPattern(),
    PatternID.CURSED_CHAIN: CursedChainPattern(),
}
