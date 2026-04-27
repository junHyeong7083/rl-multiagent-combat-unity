"""비교군 FSM NPC — 탱커 / 힐러 / 서포터 (유클리드)

원 그대로 동일한 BossRaidEnv.step() API 사용. 이동 판단만 유클리드 거리 기반.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

from .config import BossActionID, PartyRole, PatternID

if TYPE_CHECKING:
    from .env import BossRaidEnv


def _euclid(ax, ay, bx, by) -> float:
    return math.hypot(ax - bx, ay - by)


class FSMNpcPolicy:
    """하나의 NPC를 제어하는 FSM 정책."""

    def __init__(self, env: "BossRaidEnv", uid: int):
        self.env = env
        self.uid = uid

    def act(self) -> int:
        env = self.env
        u = env.units[self.uid]
        if not u.alive:
            return int(BossActionID.STAY)

        # 위험 영역 위면 회피 우선
        if self._in_any_danger(u):
            safe_move = self._find_safe_move(u)
            if safe_move is not None:
                return safe_move

        # 기믹 우선순위
        mech = self._mechanic_action(u)
        if mech is not None:
            return mech

        # 역할별 로직
        if u.role == PartyRole.TANK:
            return self._tank_logic(u)
        elif u.role == PartyRole.HEALER:
            return self._healer_logic(u)
        elif u.role == PartyRole.SUPPORT:
            return self._support_logic(u)
        return int(BossActionID.ATTACK_BASIC)

    # ──────── 유틸 ────────

    def _in_any_danger(self, u) -> bool:
        for tg in self.env.boss.telegraphs:
            if tg.contains((u.x, u.y)):
                return True
        return False

    def _find_safe_move(self, u):
        env = self.env
        for act, dx, dy in (
            (BossActionID.MOVE_UP, 0, -u.move_speed),
            (BossActionID.MOVE_DOWN, 0, u.move_speed),
            (BossActionID.MOVE_LEFT, -u.move_speed, 0),
            (BossActionID.MOVE_RIGHT, u.move_speed, 0),
        ):
            nx, ny = u.x + dx, u.y + dy
            if not (u.radius <= nx <= env.config.map_width - u.radius and
                    u.radius <= ny <= env.config.map_height - u.radius):
                continue
            safe = True
            for tg in env.boss.telegraphs:
                if tg.contains((nx, ny)):
                    safe = False; break
            if safe:
                return int(act)
        return None

    def _mechanic_action(self, u):
        env = self.env
        for tg in env.boss.telegraphs:
            if tg.pattern_id == PatternID.MARK and tg.target_unit_ids:
                mark_uid = tg.target_unit_ids[0]
                if mark_uid == u.uid:
                    others = [x for x in env.units.values() if x.uid != u.uid and x.alive]
                    if others:
                        avg_x = sum(x.x for x in others) / len(others)
                        avg_y = sum(x.y for x in others) / len(others)
                        return self._move_away(u, avg_x, avg_y)
                else:
                    mu = env.units.get(mark_uid)
                    if mu:
                        return self._move_away(u, mu.x, mu.y)

            if tg.pattern_id == PatternID.CROSS_INFERNO and tg.target_unit_ids:
                safe_quads = tg.target_unit_ids
                cx, cy = env.config.map_width / 2.0, env.config.map_height / 2.0
                targets = []
                for q in safe_quads:
                    tx = cx * 0.5 if (q & 1) == 0 else cx + cx * 0.5
                    ty = cy * 0.5 if (q & 2) == 0 else cy + cy * 0.5
                    targets.append((tx, ty))
                if targets:
                    best = min(targets, key=lambda p: _euclid(u.x, u.y, p[0], p[1]))
                    return self._move_toward(u, best[0], best[1])

        # 체인 유지
        if u.chain_turns > 0 and u.chained_with is not None:
            p = env.units.get(u.chained_with)
            if p:
                d = _euclid(u.x, u.y, p.x, p.y)
                if d > env.config.pat_chain_max_distance - 0.5:
                    return self._move_toward(u, p.x, p.y)

        # 스태거: 전원 공격 전환
        if env.boss.stagger_active:
            if u.role == PartyRole.TANK and u.cooldowns.get(int(BossActionID.TAUNT), 0) <= 0:
                return int(BossActionID.TAUNT)
            if env._boss_dist(u.x, u.y) <= u.attack_range:
                if u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) <= 0:
                    return int(BossActionID.ATTACK_SKILL)
                return int(BossActionID.ATTACK_BASIC)
            return self._move_toward_boss(u)
        return None

    def _move_toward(self, u, tx, ty):
        dx = tx - u.x; dy = ty - u.y
        if abs(dx) >= abs(dy):
            return int(BossActionID.MOVE_RIGHT) if dx > 0 else int(BossActionID.MOVE_LEFT)
        return int(BossActionID.MOVE_DOWN) if dy > 0 else int(BossActionID.MOVE_UP)

    def _move_away(self, u, tx, ty):
        return self._move_toward(u, u.x + (u.x - tx), u.y + (u.y - ty))

    def _move_toward_boss(self, u):
        return self._move_toward(u, self.env.boss.x, self.env.boss.y)

    # ──────── 탱커 ────────

    def _tank_logic(self, u):
        env = self.env
        top = env.boss.top_aggro_uid()
        if top != u.uid and u.cooldowns.get(int(BossActionID.TAUNT), 0) <= 0:
            return int(BossActionID.TAUNT)
        for tg in env.boss.telegraphs:
            if tg.contains((u.x, u.y)) and u.cooldowns.get(int(BossActionID.GUARD), 0) <= 0:
                return int(BossActionID.GUARD)
        if env._boss_dist(u.x, u.y) > u.attack_range:
            return self._move_toward_boss(u)
        return int(BossActionID.ATTACK_BASIC)

    # ──────── 힐러 ────────

    def _healer_logic(self, u):
        env = self.env
        for x in env.units.values():
            if not x.alive: continue
            if (x.hp / max(1, x.max_hp)) < 0.4:
                if _euclid(x.x, x.y, u.x, u.y) <= u.attack_range:
                    if u.cooldowns.get(int(BossActionID.HEAL), 0) <= 0:
                        return int(BossActionID.HEAL)
                else:
                    return self._move_toward(u, x.x, x.y)
        for x in env.units.values():
            if x.marked_turns > 0 and _euclid(x.x, x.y, u.x, u.y) <= u.attack_range + 0.5:
                if u.cooldowns.get(int(BossActionID.CLEANSE), 0) <= 0:
                    return int(BossActionID.CLEANSE)
        others = [x for x in env.units.values() if x.uid != u.uid and x.alive]
        if others:
            avg_x = sum(x.x for x in others) / len(others)
            avg_y = sum(x.y for x in others) / len(others)
            if _euclid(u.x, u.y, avg_x, avg_y) > 4.0:
                return self._move_toward(u, avg_x, avg_y)
        if env._boss_dist(u.x, u.y) <= u.attack_range:
            return int(BossActionID.ATTACK_BASIC)
        return int(BossActionID.STAY)

    # ──────── 서포터 ────────

    def _support_logic(self, u):
        env = self.env
        tank = next((x for x in env.units.values() if x.role == PartyRole.TANK and x.alive), None)
        if tank and (tank.hp / max(1, tank.max_hp)) < 0.6:
            if _euclid(tank.x, tank.y, u.x, u.y) <= u.attack_range + 0.5:
                if u.cooldowns.get(int(BossActionID.BUFF_SHIELD), 0) <= 0:
                    return int(BossActionID.BUFF_SHIELD)
            else:
                return self._move_toward(u, tank.x, tank.y)
        if env.boss.grog_turns > 0:
            dealer = env.units[env.config.player_slot]
            if dealer.alive and _euclid(dealer.x, dealer.y, u.x, u.y) <= u.attack_range + 0.5:
                if u.cooldowns.get(int(BossActionID.BUFF_ATK), 0) <= 0:
                    return int(BossActionID.BUFF_ATK)
        if env._boss_dist(u.x, u.y) <= u.attack_range:
            if u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) <= 0:
                return int(BossActionID.ATTACK_SKILL)
            return int(BossActionID.ATTACK_BASIC)
        return self._move_toward_boss(u)
