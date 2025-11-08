import numpy as np


class CombatSelfPlay5v5Env:
    """
    격자형 5v5 자가대전 환경.
    - 이동: 8방향 + 정지 = 9개
    - 공격: 1개 (총 10개 액션)
    - 즉발 판정이지만 사거리/산개/명중률 반영
    - 재장전 쿨다운은 유닛별로 랜덤 부여
    - 장애물/시야차단(LOS) 옵션 포함
    관측:
      팀 관측(team_obs): A와 B의 유닛 상태를 모두 평탄화한 벡터
      유닛 관측(unit_obs): 팀 관측 + 선택적 로컬 파생치(여기서는 팀 관측 그대로 사용)
    """
    ACT_STAY = 0
    ACT_N  = 1
    ACT_NE = 2
    ACT_E  = 3
    ACT_SE = 4
    ACT_S  = 5
    ACT_SW = 6
    ACT_W  = 7
    ACT_NW = 8
    ACT_ATTACK = 9

    ACT_MOVE_DIRS = {
        ACT_N:  (0, 1),
        ACT_NE: (1, 1),
        ACT_E:  (1, 0),
        ACT_SE: (1, -1),
        ACT_S:  (0, -1),
        ACT_SW: (-1, -1),
        ACT_W:  (-1, 0),
        ACT_NW: (-1, 1),
    }

    def __init__(self, width=32, height=32, n_per_team=5, max_steps=180,
                 seed=0, use_obstacles=True, obstacle_rate=0.06):
        self.width = width
        self.height = height
        self.n = n_per_team
        self.max_steps = max_steps
        self.use_obstacles = use_obstacles
        self.obstacle_rate = obstacle_rate

        self.rng = np.random.default_rng(seed)
        self.t = 0

        # [x, y, hp, fx, fy, cd, base_cd]
        self.A = np.zeros((self.n, 7), dtype=np.int32)
        self.B = np.zeros((self.n, 7), dtype=np.int32)

        # 전장
        self.blocks = np.zeros((self.width, self.height), dtype=np.int8)

        # 파라미터
        self.hp_init = 5
        self.min_range = 1
        self.max_range = 6
        self.base_cooldown = 3  # 유닛별 기본 쿨다운에 더해짐
        self.hit_prob = 0.55
        self.flank_bonus = 0.15  # 측면/후방 보정

        self.reset()

    # ---------------- Utils ----------------

    def _in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _blocked(self, x, y):
        return self.blocks[x, y] == 1

    def _bresenham_clear(self, x0, y0, x1, y1):
        # x0,y0 → x1,y1 사이에 장애물 있는지 (끝점 제외)
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            if (x, y) != (x0, y0) and (x, y) != (x1, y1):
                if self._blocked(x, y):
                    return False
            if (x, y) == (x1, y1):
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return True

    def _distance(self, x0, y0, x1, y1):
        return max(abs(x1 - x0), abs(y1 - y0))

    # ---------------- Reset ----------------

    def reset(self):
        self.t = 0

        # 전장 리셋
        self.blocks[:] = 0
        if self.use_obstacles:
            mask = self.rng.random((self.width, self.height)) < self.obstacle_rate
            self.blocks[mask] = 1

        def spawn_team(left_side=True):
            team = np.zeros((self.n, 7), dtype=np.int32)
            for i in range(self.n):
                if left_side:
                    x = self.rng.integers(1, self.width // 3)
                else:
                    x = self.rng.integers(self.width - self.width // 3, self.width - 1)
                y = self.rng.integers(1, self.height - 1)
                team[i, 0] = x
                team[i, 1] = y
                team[i, 2] = self.hp_init
                # 초기 바라보는 방향: 중앙 쪽
                team[i, 3] = 1 if left_side else -1
                team[i, 4] = 0
                # 쿨다운
                base_cd = self.base_cooldown + int(self.rng.integers(0, 3))
                team[i, 5] = 0
                team[i, 6] = base_cd
            return team

        self.A = spawn_team(left_side=True)
        self.B = spawn_team(left_side=False)

        return self._obs_team(self.A, self.B), self._obs_team(self.B, self.A)

    # ---------------- Attack core ----------------

    def _attack_targets(self, me, other):
        # 각 유닛이 조준 방향으로 공격할 때 맞을 수 있는 타겟을 찾고 명중 확률/산개를 적용
        hits = []
        for i in range(self.n):
            x, y, hp, fx, fy, cd, base_cd = me[i]
            if hp <= 0 or cd > 0:
                continue
            tx, ty = x + fx, y + fy
            # 사거리와 산개 반영
            best_j = None
            best_d = None
            for j in range(self.n):
                ox, oy, ohp, *_ = other[j]
                if ohp <= 0:
                    continue
                d = self._distance(x, y, ox, oy)
                if d == 0 or d > self.max_range:
                    continue
                vx = np.sign(ox - x)
                vy = np.sign(oy - y)
                if vx == fx and vy == fy:
                    if self._bresenham_clear(x, y, ox, oy):
                        if best_d is None or d < best_d:
                            best_d = d
                            best_j = j
            if best_j is None:
                continue

            ox, oy, ohp, ofx, ofy, ocd, obase = other[best_j]
            # 측면/후방 보정(상대의 바라보는 방향 기준)
            flanking = (ofx, ofy) != (-fx, -fy)
            p = self.hit_prob + (self.flank_bonus if flanking else 0.0)
            if self.rng.random() < p:
                hits.append({
                    "from": [int(x), int(y)],
                    "to": [int(ox), int(oy)],
                    "hit": True
                })
                me[i, 5] = base_cd  # 쿨다운 리셋
            else:
                hits.append({
                    "from": [int(x), int(y)],
                    "to": [int(ox), int(oy)],
                    "hit": False
                })
                me[i, 5] = max(1, base_cd // 2)

        return hits

    # ---------------- Step ----------------

    def step(self, actA: np.ndarray, actB: np.ndarray):
        assert actA.shape == (self.n,) and actB.shape == (self.n,)
        self.t += 1

        shots = []

        # 쿨다운 감소
        for i in range(self.n):
            if self.A[i, 5] > 0:
                self.A[i, 5] -= 1
            if self.B[i, 5] > 0:
                self.B[i, 5] -= 1

        # 이동 처리
        def apply_move(team, other, acts):
            for i in range(self.n):
                x, y, hp, fx, fy, cd, base_cd = team[i]
                if hp <= 0:
                    continue
                a = int(acts[i])
                if a == self.ACT_STAY:
                    pass
                elif a in self.ACT_MOVE_DIRS:
                    dx, dy = self.ACT_MOVE_DIRS[a]
                    nx, ny = x + dx, y + dy
                    if self._in_bounds(nx, ny) and not self._blocked(nx, ny):
                        team[i, 0] = nx
                        team[i, 1] = ny
                        team[i, 3] = dx  # facing
                        team[i, 4] = dy
                elif a == self.ACT_ATTACK:
                    # 가장 가까운 생존 적을 향해 조준만 변경(위치 이동 없음)
                    best = None
                    best_d = None
                    for j in range(self.n):
                        ox, oy, ohp, *_ = other[j]
                        if ohp <= 0:
                            continue
                        d = self._distance(x, y, ox, oy)
                        if best_d is None or d < best_d:
                            best_d = d
                            best = (ox, oy)
                    if best is not None:
                        vx = int(np.sign(best[0] - x))
                        vy = int(np.sign(best[1] - y))
                        if vx != 0 or vy != 0:
                            team[i, 3] = vx
                            team[i, 4] = vy

        apply_move(self.A, self.B, actA)
        apply_move(self.B, self.A, actB)

        # 공격 처리: 사거리/산개/명중률
        hitsA = self._attack_targets(self.A, self.B)
        hitsB = self._attack_targets(self.B, self.A)
        shots.extend([{"team": "A", **h} for h in hitsA])
        shots.extend([{"team": "B", **h} for h in hitsB])

        # 피해 적용 및 보상
        rA = 0.0
        rB = 0.0

        # 보상 셰이핑: 엄폐 인접 생존, 유효사거리 내 조준
        def shaping_reward(team, other):
            bonus = 0.0
            for i in range(self.n):
                x, y, hp, fx, fy, cd, base_cd = team[i]
                if hp <= 0:
                    continue
                # 엄폐 인접
                has_cover = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height and self._blocked(nx, ny):
                            has_cover = True
                            break
                    if has_cover:
                        break
                if has_cover:
                    bonus += 0.03
                # 유효사거리 내 조준
                aligned_in_range = False
                for j in range(self.n):
                    ox, oy, ohp, *_ = other[j]
                    if ohp <= 0:
                        continue
                    d = self._distance(x, y, ox, oy)
                    if d == 0 or d > self.max_range:
                        continue
                    vx = np.sign(ox - x)
                    vy = np.sign(oy - y)
                    if vx == team[i, 3] and vy == team[i, 4]:
                        aligned_in_range = True
                        break
                if aligned_in_range:
                    bonus += 0.02
            return bonus

        rA += shaping_reward(self.A, self.B)
        rB += shaping_reward(self.B, self.A)

        # 명중 피해/보상
        def apply_hits(hits, me, other, me_is_A: bool):
            nonlocal rA, rB
            for h in hits:
                if not h["hit"]:
                    continue
                tx, ty = h["to"]
                # 타겟 식별(가장 가까운 좌표 매칭)
                best_j = None
                best_d = None
                for j in range(self.n):
                    ox, oy, ohp, *_ = other[j]
                    if ohp <= 0:
                        continue
                    d = self._distance(tx, ty, ox, oy)
                    if best_d is None or d < best_d:
                        best_d = d
                        best_j = j
                if best_j is None:
                    continue
                # 측면 보정
                mx, my, mhp, mfx, mfy, mcd, mbase = me[0]  # 아무 유닛의 facing 참고용
                fx, fy = mfx, mfy
                ox, oy, ohp, ofx, ofy, ocd, obase = other[best_j]
                flanking = (ofx, ofy) != (-fx, -fy)

                other[best_j, 2] -= 1
                if me_is_A:
                    rA += 1.0 + (0.05 if flanking else 0.0)
                    rB -= 1.0
                else:
                    rB += 1.0 + (0.05 if flanking else 0.0)
                    rA -= 1.0

        apply_hits(hitsA, self.A, self.B, True)
        apply_hits(hitsB, self.B, self.A, False)

        # 종료 조건
        aliveA = int(np.sum(self.A[:, 2] > 0))
        aliveB = int(np.sum(self.B[:, 2] > 0))
        done = False
        outcome = None
        if self.t >= self.max_steps or aliveA == 0 or aliveB == 0:
            done = True
            if aliveA > aliveB:
                rA += 5.0
                rB -= 5.0
                outcome = "A_wipe" if aliveB == 0 else "timeout"
            elif aliveB > aliveA:
                rB += 5.0
                rA -= 5.0
                outcome = "B_wipe" if aliveA == 0 else "timeout"
            else:
                outcome = "draw"

        info = {
            "shots": shots,
            "blocks": self.blocks.copy(),
            "aliveA": aliveA,
            "aliveB": aliveB,
            "outcome": outcome,
        }
        return self._obs_team(self.A, self.B), self._obs_team(self.B, self.A), float(rA), float(rB), done, info

    # -------- Observations --------

    def _obs_team(self, me, other):
        # x,y,hp,fx,fy,cd,base_cd -> 7개
        return np.concatenate([me.flatten(), other.flatten()]).astype(np.int32)

    def get_team_obs_dim(self):
        return (7 * self.n * 2)

    def sample_actions(self):
        return self.rng.integers(0, 10, size=self.n, dtype=np.int32)
