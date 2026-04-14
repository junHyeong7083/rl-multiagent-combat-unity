"""보상 계산기 — 공통 / 역할별 / 패턴별"""
from __future__ import annotations
from typing import Dict, TYPE_CHECKING

from .config import BossConfig, PartyRole, PatternID

if TYPE_CHECKING:
    from .env import BossRaidEnv


class RewardComputer:
    def __init__(self, cfg: BossConfig):
        self.cfg = cfg

    def compute(self, env: "BossRaidEnv") -> Dict[str, float]:
        cfg = self.cfg
        out: Dict[str, float] = {}
        player_uid = cfg.player_slot
        player = env.units[player_uid]

        # 보스 HP 변화 (전 스텝 대비) — 간이: events의 damage 합으로 계산
        # (더 정확하려면 이전 HP 저장 필요; 여기서는 events 기반)
        boss_damage_total = 0
        for uid, events in env.step_events.items():
            for e in events:
                if e.get("type") == "damage":
                    boss_damage_total += e.get("amount", 0)

        for uid, u in env.units.items():
            aid = f"p{uid}"
            events = env.step_events.get(uid, [])
            r = 0.0

            # ── 공통 ──
            # 내가 준 데미지
            dmg_dealt = sum(e.get("amount", 0) for e in events if e.get("type") == "damage")
            r += dmg_dealt * cfg.rw_boss_damage_per_hp

            # 플레이어 생존
            if player.alive:
                r += cfg.rw_player_alive_step
            else:
                if any(e.get("type") == "death" for e in env.step_events.get(player_uid, [])):
                    r += cfg.rw_player_death

            # 자신 사망
            if any(e.get("type") == "death" for e in events):
                r += cfg.rw_npc_death

            # 기믹 성공/실패
            for e in events:
                if e.get("type") == "mechanic_success":
                    r += cfg.rw_mechanic_success
                elif e.get("type") == "mechanic_fail":
                    r += cfg.rw_mechanic_fail
                elif e.get("type") == "stagger_success":
                    r += cfg.rw_stagger_success
                elif e.get("type") == "stagger_fail":
                    r += cfg.rw_mechanic_fail
                elif e.get("type") == "invalid_action":
                    r += cfg.rw_invalid_action
                elif e.get("type") == "damage_taken":
                    r += cfg.rw_danger_hit
                elif e.get("type") == "phase_clear":
                    r += cfg.rw_phase_clear

            # 텔레그래프 회피 보상 (위험 타일 위 아닐 때 활성 텔레그래프가 있으면)
            danger = set()
            for tg in env.boss.telegraphs:
                danger |= tg.danger_tiles
            if danger and (u.x, u.y) not in danger:
                r += cfg.rw_telegraph_dodge * 0.1  # 작은 값 매 턴

            # ── 역할별 ──
            if u.role == PartyRole.TANK:
                r += self._tank_reward(env, u, events)
            elif u.role == PartyRole.HEALER:
                r += self._healer_reward(env, u, events)
            elif u.role == PartyRole.SUPPORT:
                r += self._support_reward(env, u, events)
            # 딜러(플레이어)는 외부 조작이지만 계산은 해둔다 (학습 시 더미)

            # ── 패턴별 특수 보상 ──
            r += self._pattern_rewards(env, u, events)

            # ── 종료 보상 ──
            if env.done:
                if env.victory:
                    r += cfg.rw_boss_kill
                if env.wipe:
                    r += cfg.rw_wipe

            out[aid] = r

        return out

    # ──────────────── 역할별 ────────────────

    def _tank_reward(self, env, u, events) -> float:
        cfg = self.cfg
        r = 0.0
        top = env.boss.top_aggro_uid()
        if top == u.uid:
            r += cfg.rw_tank_aggro_hold
        elif u.alive:
            r += cfg.rw_tank_aggro_lose * 0.3  # 약한 패널티
        # 도발 적시 사용
        if any(e.get("type") == "taunt" for e in events):
            # 직전 어그로 1위가 내가 아니었다면 good taunt
            r += cfg.rw_tank_taunt_good
        # 보스 근접
        if env._boss_dist(u.x, u.y) <= 2:
            r += cfg.rw_tank_close_boss
        # 플레이어 보호
        player = env.units[cfg.player_slot]
        if env._boss_dist(u.x, u.y) < env._boss_dist(player.x, player.y):
            r += cfg.rw_tank_guard_player
        return r

    def _healer_reward(self, env, u, events) -> float:
        cfg = self.cfg
        r = 0.0
        # 힐
        for e in events:
            if e.get("type") == "heal":
                amount = e.get("amount", 0)
                r += amount * cfg.rw_heal_per_hp
                # critical 대상이면 추가
                t = env.units.get(e.get("target"))
                if t and (t.hp / max(1, t.max_hp)) < 0.3:
                    r += cfg.rw_heal_critical
        # 스태거 중 공격
        if env.boss.stagger_active and any(e.get("type") == "damage" for e in events):
            r += cfg.rw_healer_stagger_atk
        # 파티 중앙 위치
        avg_x = sum(ux.x for ux in env.units.values() if ux.alive) / max(1, sum(1 for ux in env.units.values() if ux.alive))
        avg_y = sum(ux.y for ux in env.units.values() if ux.alive) / max(1, sum(1 for ux in env.units.values() if ux.alive))
        dist_center = abs(u.x - avg_x) + abs(u.y - avg_y)
        if dist_center < 3:
            r += cfg.rw_healer_central
        return r

    def _support_reward(self, env, u, events) -> float:
        cfg = self.cfg
        r = 0.0
        for e in events:
            if e.get("type") == "buff":
                r += cfg.rw_buff_hit
                # 탱커 선제 실드 (탱커에게 shield 버프, 텔레그래프 활성 중)
                t = env.units.get(e.get("target"))
                if t and t.role == PartyRole.TANK and e.get("kind") == "shield":
                    if env.boss.telegraphs:
                        r += cfg.rw_buff_tank_pre
                # 그로기 중 딜러 공버프
                if t and t.role == PartyRole.DEALER and e.get("kind") == "atk":
                    if env.boss.grog_turns > 0:
                        r += cfg.rw_buff_dealer_grog
        # 스태거 중 공격
        if env.boss.stagger_active and any(e.get("type") == "damage" for e in events):
            r += cfg.rw_support_stagger_atk
        return r

    # ──────────────── 패턴별 특수 ────────────────

    def _pattern_rewards(self, env, u, events) -> float:
        cfg = self.cfg
        r = 0.0
        # MARK: 표식/비표식 분리
        for tg in env.boss.telegraphs:
            if tg.pattern_id == PatternID.MARK and tg.target_unit_ids:
                mark_uid = tg.target_unit_ids[0]
                if mark_uid == u.uid:
                    # 내가 표식 → 파티에서 이탈해야 함
                    others = [x for x in env.units.values() if x.uid != u.uid and x.alive]
                    if others:
                        min_d = min(max(abs(x.x - u.x), abs(x.y - u.y)) for x in others)
                        if min_d >= 5:
                            r += cfg.rw_mark_carrier_spread * 0.3
                else:
                    # 나는 비표식 → 표식자 반대로 이동
                    if mark_uid in env.units:
                        mu = env.units[mark_uid]
                        d = max(abs(mu.x - u.x), abs(mu.y - u.y))
                        if d >= 5:
                            r += cfg.rw_mark_other_spread * 0.3

            elif tg.pattern_id == PatternID.STAGGER:
                # 보스 근접 집결 보상
                if env._boss_dist(u.x, u.y) <= 2:
                    r += cfg.rw_stagger_gather

            elif tg.pattern_id == PatternID.CROSS_INFERNO and tg.target_unit_ids:
                safe_quads = tg.target_unit_ids
                q = env._quadrant(u.x, u.y)
                if q in safe_quads:
                    r += cfg.rw_cross_gather * 0.05
                else:
                    r += cfg.rw_cross_split * 0.05

        # 체인 유지
        if u.chain_turns > 0 and u.chained_with is not None:
            partner = env.units.get(u.chained_with)
            if partner:
                d = max(abs(u.x - partner.x), abs(u.y - partner.y))
                if d <= 3:
                    r += cfg.rw_chain_hold
        return r
