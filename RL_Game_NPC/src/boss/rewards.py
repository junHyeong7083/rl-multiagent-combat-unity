"""보상 계산기 v26 — B안 관측과 일관된 협동 패턴 shaping

핵심 변경 (v20 대비):
  - MARK/CROSS 이분법 → 거리 그라디언트
  - CHAIN wind_up 3턴 접근 shaping 추가
  - SEAL: NPC 배정된 spot 기반 보상 (tg.extra["npc_spots"])
  - STAGGER: stagger_contribute 이벤트 연결 (실제 게이지 기여당 보상)
  - 페이즈 전이 / SEAL 중 역할 보상 감쇠 (협동 보상이 묻히지 않게)

원칙은 유지:
  위험 영역 안(발동 임박) → 도망만
  위험 영역 밖 → 역할 수행
"""
from __future__ import annotations
import math
from typing import Dict, TYPE_CHECKING

from .config import BossConfig, PartyRole, PatternID

if TYPE_CHECKING:
    from .env import BossRaidEnv


def _dist(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


# ── CROSS 사분면 중심 (rewards/obs 양쪽에서 사용) ──
def _quad_centers(cfg: BossConfig) -> Dict[int, tuple]:
    cx = cfg.map_width / 2.0
    cy = cfg.map_height / 2.0
    return {
        0: (cx * 0.5, cy * 0.5),
        1: (cx * 1.5, cy * 0.5),
        2: (cx * 0.5, cy * 1.5),
        3: (cx * 1.5, cy * 1.5),
    }


class RewardComputer:
    def __init__(self, cfg: BossConfig):
        self.cfg = cfg

    def compute(self, env: "BossRaidEnv") -> Dict[str, float]:
        cfg = self.cfg
        out: Dict[str, float] = {}
        player = env.units[cfg.player_slot]

        # ── 페이즈 전이 / SEAL 중: 역할 보상 감쇠 배율 ──
        # 이 구간엔 딜/힐/버프보다 "자리 잡기"가 중요 → 역할 보상 * 0.3
        seal_in_progress = any(tg.pattern_id == PatternID.SEAL_BREAK
                               for tg in env.boss.telegraphs)
        role_scale = 0.3 if (seal_in_progress or env.boss.invuln_turns > 0) else 1.0

        for uid, u in env.units.items():
            aid = f"p{uid}"
            events = env.step_events.get(uid, [])
            r = 0.0

            if not u.alive:
                out[aid] = r
                continue

            boss_dist = env._boss_dist(u.x, u.y)
            player_dist = _dist(u.x, u.y, player.x, player.y)

            # ── 위험 영역 판정 (SEAL 제외) ──
            in_danger = False
            if env.boss.telegraphs:
                in_danger = any(
                    tg.contains((u.x, u.y))
                    for tg in env.boss.telegraphs
                    if tg.pattern_id != PatternID.SEAL_BREAK
                    and tg.turns_remaining <= 2
                )

            early_warning = False
            if env.boss.telegraphs and not in_danger:
                early_warning = any(
                    tg.contains((u.x, u.y))
                    for tg in env.boss.telegraphs
                    if tg.pattern_id != PatternID.SEAL_BREAK
                    and tg.turns_remaining > 2
                )
                if early_warning:
                    r += -1.0

            # ═══════════════════════════════════════
            # 위험 모드: 도망만
            # ═══════════════════════════════════════
            if in_danger:
                r += -5.0
                for e in events:
                    if e.get("type") == "damage_taken":
                        r += -8.0
                    elif e.get("type") == "death":
                        r += -15.0

            # ═══════════════════════════════════════
            # 안전 모드: 역할 수행 (× role_scale)
            # ═══════════════════════════════════════
            else:
                if env.boss.telegraphs:
                    r += 1.0 * role_scale

                # 플레이어 보좌 (공통)
                if player_dist <= 4.0:
                    r += 1.5 * role_scale
                elif player_dist <= 7.0:
                    r += 0.3 * role_scale
                elif player_dist > 10.0:
                    r += -0.5

                # 탱커
                if u.role == PartyRole.TANK:
                    if boss_dist <= 2.0:
                        r += 2.0 * role_scale
                    if env.boss.top_aggro_uid() == u.uid:
                        r += 1.5 * role_scale
                    else:
                        r += -1.0 * role_scale
                    if any(e.get("type") == "taunt" for e in events):
                        r += 2.0
                    if boss_dist < env._boss_dist(player.x, player.y):
                        r += 1.0 * role_scale
                    dmg = sum(e.get("amount", 0) for e in events if e.get("type") == "damage")
                    r += dmg * 0.1 * role_scale

                # 힐러
                elif u.role == PartyRole.HEALER:
                    for e in events:
                        if e.get("type") == "heal":
                            amount = e.get("amount", 0)
                            r += amount * 0.5
                            target = env.units.get(e.get("target"))
                            if target and (target.hp / max(1, target.max_hp)) < 0.3:
                                r += 3.0
                            if e.get("target") == cfg.player_slot:
                                r += 2.0
                    for x in env.units.values():
                        if x.uid == u.uid or not x.alive:
                            continue
                        if (x.hp / max(1, x.max_hp)) < 0.5:
                            if _dist(u.x, u.y, x.x, x.y) <= u.attack_range:
                                r += 0.5 * role_scale
                                break
                    if _dist(u.x, u.y, player.x, player.y) <= u.attack_range:
                        r += 0.5 * role_scale

                # 서포터
                elif u.role == PartyRole.SUPPORT:
                    for e in events:
                        if e.get("type") == "buff":
                            r += 2.5
                            target = env.units.get(e.get("target"))
                            if target and target.role == PartyRole.TANK:
                                r += 1.5
                            if target and target.role == PartyRole.DEALER:
                                r += 2.0
                    dmg = sum(e.get("amount", 0) for e in events if e.get("type") == "damage")
                    r += dmg * 0.2 * role_scale
                    if boss_dist <= u.attack_range + 1.0:
                        r += 0.5 * role_scale

                # 딜러 (FSM/player)
                elif u.role == PartyRole.DEALER:
                    dmg = sum(e.get("amount", 0) for e in events if e.get("type") == "damage")
                    r += dmg * 0.15

                # 안전 모드 피격/사망
                for e in events:
                    if e.get("type") == "damage_taken":
                        r += -4.0
                    elif e.get("type") == "death":
                        r += -15.0

            # ═══════════════════════════════════════
            # 항상 적용
            # ═══════════════════════════════════════

            # 생존 보상 (전투 참여 중일 때만)
            engaged = player_dist <= 8.0
            if engaged:
                r += 0.5
                alive_count = sum(1 for x in env.units.values() if x.alive)
                r += alive_count * 0.15

            if player.alive:
                r += 0.3
            else:
                r += -5.0

            for e in env.step_events.get(cfg.player_slot, []):
                if e.get("type") == "damage_taken":
                    r += -1.5
                elif e.get("type") == "death":
                    r += -10.0

            # 기믹 결과 이벤트
            for e in events:
                t = e.get("type")
                if t == "mechanic_success":
                    r += 20.0
                elif t == "mechanic_fail":
                    r += -15.0
                elif t == "stagger_success":
                    r += 30.0
                elif t == "stagger_fail":
                    r += -20.0
                elif t == "phase_clear":
                    r += 15.0
                elif t == "seal_success":
                    r += 50.0
                elif t == "seal_fail":
                    r += -30.0
                elif t == "seal_holding":
                    r += 5.0
                elif t == "stagger_contribute":
                    r += e.get("amount", 0) * cfg.rw_stagger_contribution

            # ═══════════════════════════════════════
            # 협동 패턴 shaping (이분법 제거)
            # ═══════════════════════════════════════

            # 스태거 집결
            if env.boss.stagger_active:
                if boss_dist <= cfg.stagger_gather_radius:
                    r += cfg.rw_stagger_gather
                # 힐러도 스태거 타임엔 딜 기여 보너스
                if u.role == PartyRole.HEALER:
                    dmg_h = sum(e.get("amount", 0) for e in events if e.get("type") == "damage")
                    r += dmg_h * cfg.rw_healer_stagger_atk

            for tg in env.boss.telegraphs:

                # MARK — 그라디언트: 거리/이상거리 비율
                if tg.pattern_id == PatternID.MARK and tg.target_unit_ids:
                    mark_uid = tg.target_unit_ids[0]
                    ideal = cfg.pat_mark_escape_distance
                    urgency = 1.0 - tg.turns_remaining / max(1, tg.total_wind_up)
                    if mark_uid == u.uid:
                        # 내가 표식자: 가장 가까운 아군과 멀수록 +
                        others = [x for x in env.units.values()
                                  if x.uid != u.uid and x.alive]
                        if others:
                            min_d = min(_dist(x.x, x.y, u.x, u.y) for x in others)
                            progress = min(min_d / ideal, 1.0)   # 0~1
                            r += progress * cfg.rw_mark_carrier_spread * (0.3 + urgency)
                    else:
                        # 내가 아님: 표식자와 멀수록 +
                        if mark_uid in env.units:
                            mu = env.units[mark_uid]
                            d = _dist(mu.x, mu.y, u.x, u.y)
                            progress = min(d / ideal, 1.0)
                            r += progress * cfg.rw_mark_other_spread * (0.3 + urgency)

                # CROSS — 그라디언트: 안전 사분면까지 근접도
                elif tg.pattern_id == PatternID.CROSS_INFERNO:
                    safe_quads = tg.target_unit_ids
                    q = env._quadrant(u.x, u.y)
                    urgency = 1.0 - tg.turns_remaining / max(1, tg.total_wind_up)
                    if q in safe_quads:
                        r += cfg.rw_cross_gather * (0.4 + 0.6 * urgency)
                    else:
                        # 가장 가까운 안전 사분면 중심까지 거리 → 접근하면 패널티 감쇠
                        centers = _quad_centers(cfg)
                        best_d = 1e9
                        for sq in safe_quads:
                            if sq in centers:
                                cx_s, cy_s = centers[sq]
                                best_d = min(best_d, _dist(u.x, u.y, cx_s, cy_s))
                        max_d = math.hypot(cfg.map_width, cfg.map_height)
                        far_ratio = min(best_d / max_d, 1.0)         # 1=아주 멀리, 0=경계
                        r += cfg.rw_cross_split * far_ratio * (0.4 + 0.6 * urgency)

                # SEAL — 배정된 spot 기반
                elif tg.pattern_id == PatternID.SEAL_BREAK:
                    spots = tg.extra.get("npc_spots", {})
                    if u.uid != cfg.player_slot and u.uid in spots:
                        si = spots[u.uid]
                        if 0 <= si < len(tg.shapes):
                            s = tg.shapes[si]
                            sx, sy = s.params["cx"], s.params["cy"]
                            d = _dist(u.x, u.y, sx, sy)
                            if d <= s.params["r"]:
                                r += 3.0                             # 내 spot 점유
                            else:
                                # 거리 shaping — 접근할수록 패널티 감쇠
                                proximity = max(0.0, 1.0 - d / 10.0)
                                r += -3.0 * (1.0 - proximity)
                    # 팀 진행도 보너스: 배정된 NPC 중 자기 spot에 있는 수
                    on_assigned = 0
                    for npc_uid, si in spots.items():
                        if npc_uid not in env.units or not env.units[npc_uid].alive:
                            continue
                        if 0 <= si < len(tg.shapes):
                            s = tg.shapes[si]
                            if _dist(env.units[npc_uid].x, env.units[npc_uid].y,
                                     s.params["cx"], s.params["cy"]) <= s.params["r"]:
                                on_assigned += 1
                    r += on_assigned * 1.5

                # CURSED_CHAIN — wind_up 3턴 접근 shaping
                elif tg.pattern_id == PatternID.CURSED_CHAIN \
                        and len(tg.target_unit_ids) >= 2 and not tg.impacted:
                    a, b = tg.target_unit_ids[:2]
                    if u.uid in (a, b):
                        partner_uid = b if u.uid == a else a
                        if partner_uid in env.units:
                            p = env.units[partner_uid]
                            d = _dist(u.x, u.y, p.x, p.y)
                            ideal = cfg.pat_chain_max_distance
                            closeness = max(0.0, 1.0 - d / (ideal * 2))  # 0 at 2*ideal, 1 touching
                            urgency = 1.0 - tg.turns_remaining / max(1, tg.total_wind_up)
                            r += closeness * urgency * cfg.rw_chain_hold

            # 체인 유지 (post-impact)
            if u.chain_turns > 0 and u.chained_with is not None:
                partner = env.units.get(u.chained_with)
                if partner:
                    d = _dist(u.x, u.y, partner.x, partner.y)
                    if d <= cfg.pat_chain_max_distance:
                        r += cfg.rw_chain_hold
                    else:
                        r += -1.5

            # 시간 패널티
            r += cfg.rw_time_penalty

            # 종료 보상
            if env.done:
                if env.victory:
                    r += 150.0
                elif env.wipe:
                    r += -100.0

            out[aid] = r

        return out
