"""비교군 FSM NPC — 탱커 / 힐러 / 서포터

RL과 동일한 BossRaidEnv.step() API에 맞춰 action_id(int)를 반환한다.
파일럿 후 튜닝 필요하지만 "상용 게임 수준의 합리적 파티 AI" 목표.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from .config import BossActionID, PartyRole, PatternID

if TYPE_CHECKING:
    from .env import BossRaidEnv, PartyUnit


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

        # 위험 타일 위면 무조건 회피 우선
        danger = set()
        for tg in env.boss.telegraphs:
            danger |= tg.danger_tiles
        if (u.x, u.y) in danger:
            safe_move = self._find_safe_move(u, danger)
            if safe_move is not None:
                return safe_move

        # 기믹 우선순위
        mech = self._mechanic_action(u, danger)
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

    # ──────── 공통 ────────

    def _find_safe_move(self, u, danger):
        env = self.env
        w, h = env.config.map_width, env.config.map_height
        for act, dx, dy in (
            (BossActionID.MOVE_UP, 0, -1),
            (BossActionID.MOVE_DOWN, 0, 1),
            (BossActionID.MOVE_LEFT, -1, 0),
            (BossActionID.MOVE_RIGHT, 1, 0),
        ):
            nx, ny = u.x + dx, u.y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if (nx, ny) in danger:
                continue
            return int(act)
        return None

    def _mechanic_action(self, u, danger):
        env = self.env
        # MARK: 내가 표식이면 파티 반대로, 아니면 반대 방향으로
        for tg in env.boss.telegraphs:
            if tg.pattern_id == PatternID.MARK and tg.target_unit_ids:
                mark_uid = tg.target_unit_ids[0]
                if mark_uid == u.uid:
                    # 파티 평균 반대 방향으로
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
                # 가장 가까운 안전 사분면 중심으로 이동
                safe_quads = tg.target_unit_ids
                cx, cy = env.config.map_width // 2, env.config.map_height // 2
                targets = []
                for q in safe_quads:
                    tx = cx // 2 if q in (0, 2) else cx + cx // 2
                    ty = cy // 2 if q in (0, 1) else cy + cy // 2
                    targets.append((tx, ty))
                if targets:
                    best = min(targets, key=lambda p: abs(u.x - p[0]) + abs(u.y - p[1]))
                    return self._move_toward(u, best[0], best[1])

        # 체인 유지
        if u.chain_turns > 0 and u.chained_with is not None:
            p = env.units.get(u.chained_with)
            if p:
                d = max(abs(u.x - p.x), abs(u.y - p.y))
                if d > 3:
                    return self._move_toward(u, p.x, p.y)

        # 스태거 중이면 전원 공격 전환
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
        else:
            return int(BossActionID.MOVE_DOWN) if dy > 0 else int(BossActionID.MOVE_UP)

    def _move_away(self, u, tx, ty):
        dx = u.x - tx; dy = u.y - ty
        if abs(dx) >= abs(dy):
            if dx >= 0:
                return int(BossActionID.MOVE_RIGHT) if u.x + 1 < self.env.config.map_width else int(BossActionID.MOVE_LEFT)
            return int(BossActionID.MOVE_LEFT) if u.x > 0 else int(BossActionID.MOVE_RIGHT)
        else:
            if dy >= 0:
                return int(BossActionID.MOVE_DOWN) if u.y + 1 < self.env.config.map_height else int(BossActionID.MOVE_UP)
            return int(BossActionID.MOVE_UP) if u.y > 0 else int(BossActionID.MOVE_DOWN)

    def _move_toward_boss(self, u):
        return self._move_toward(u, self.env.boss.x, self.env.boss.y)

    # ──────── 탱커 ────────

    def _tank_logic(self, u):
        env = self.env
        top = env.boss.top_aggro_uid()
        # 어그로 뺏김 + 도발 가능
        if top != u.uid and u.cooldowns.get(int(BossActionID.TAUNT), 0) <= 0:
            return int(BossActionID.TAUNT)
        # 텔레그래프 활성 + 내 범위 영향 → GUARD
        for tg in env.boss.telegraphs:
            if (u.x, u.y) in tg.danger_tiles and u.cooldowns.get(int(BossActionID.GUARD), 0) <= 0:
                return int(BossActionID.GUARD)
        # 보스 공격 범위 밖이면 접근
        if env._boss_dist(u.x, u.y) > u.attack_range:
            return self._move_toward_boss(u)
        # 공격
        return int(BossActionID.ATTACK_BASIC)

    # ──────── 힐러 ────────

    def _healer_logic(self, u):
        env = self.env
        # 아군 HP 40% 이하 있으면 힐
        for x in env.units.values():
            if not x.alive: continue
            if (x.hp / max(1, x.max_hp)) < 0.4:
                if abs(x.x - u.x) + abs(x.y - u.y) <= u.attack_range:
                    if u.cooldowns.get(int(BossActionID.HEAL), 0) <= 0:
                        return int(BossActionID.HEAL)
                else:
                    return self._move_toward(u, x.x, x.y)
        # 표식 해제 필요
        for x in env.units.values():
            if x.marked_turns > 0 and abs(x.x - u.x) + abs(x.y - u.y) <= u.attack_range + 1:
                if u.cooldowns.get(int(BossActionID.CLEANSE), 0) <= 0:
                    return int(BossActionID.CLEANSE)
        # 파티 중앙 유지
        others = [x for x in env.units.values() if x.uid != u.uid and x.alive]
        if others:
            avg_x = sum(x.x for x in others) / len(others)
            avg_y = sum(x.y for x in others) / len(others)
            dist = abs(u.x - avg_x) + abs(u.y - avg_y)
            if dist > 4:
                return self._move_toward(u, int(avg_x), int(avg_y))
        # 딜 참여
        if env._boss_dist(u.x, u.y) <= u.attack_range:
            return int(BossActionID.ATTACK_BASIC)
        return int(BossActionID.STAY)

    # ──────── 서포터 ────────

    def _support_logic(self, u):
        env = self.env
        # 탱커 HP 60% 이하 → 방어 버프
        tank = next((x for x in env.units.values() if x.role == PartyRole.TANK and x.alive), None)
        if tank and (tank.hp / max(1, tank.max_hp)) < 0.6:
            if abs(tank.x - u.x) + abs(tank.y - u.y) <= u.attack_range + 1:
                if u.cooldowns.get(int(BossActionID.BUFF_SHIELD), 0) <= 0:
                    return int(BossActionID.BUFF_SHIELD)
            else:
                return self._move_toward(u, tank.x, tank.y)
        # 그로기 중 → 딜러 공버프
        if env.boss.grog_turns > 0:
            dealer = env.units[env.config.player_slot]
            if dealer.alive and abs(dealer.x - u.x) + abs(dealer.y - u.y) <= u.attack_range + 1:
                if u.cooldowns.get(int(BossActionID.BUFF_ATK), 0) <= 0:
                    return int(BossActionID.BUFF_ATK)
        # 보스 공격 참여
        if env._boss_dist(u.x, u.y) <= u.attack_range:
            if u.cooldowns.get(int(BossActionID.ATTACK_SKILL), 0) <= 0:
                return int(BossActionID.ATTACK_SKILL)
            return int(BossActionID.ATTACK_BASIC)
        return self._move_toward_boss(u)
