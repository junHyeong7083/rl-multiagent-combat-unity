import numpy as np

class CombatSelfPlay5v5Env:
    ACT_STAY = 0
    ACT_MOVE_DIRS = {
        1: (0, -1), 2: (0,  1), 3: (-1, 0), 4: (1,  0),
        5: (-1,-1), 6: (1, -1), 7: (-1, 1), 8: (1,  1),
    }
    ACT_ATTACK = 9

    def __init__(self, width=24, height=16, n_per_team=5, max_steps=120,  # ★ 기본 맵 키움
                 hp=3, reload_steps=5, seed=None):
        self.width = width
        self.height = height
        self.n = n_per_team
        self.max_steps = max_steps
        self.max_hp = hp
        self.reload_steps = reload_steps
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        self.t = 0
        self.A = self._spawn_side(left=True)
        self.B = self._spawn_side(left=False)
        return self._obs_team(self.A, self.B), self._obs_team(self.B, self.A)

    def _spawn_side(self, left=True):
        xs = set()
        agents = []
        tries = 0
        while len(agents) < self.n and tries < 2000:
            tries += 1
            if left:
                x = self.rng.randint(0, max(1, self.width//3))
            else:
                x = self.rng.randint(self.width - max(1, self.width//3), self.width)
            y = self.rng.randint(0, self.height)
            if (x,y) in xs:
                continue
            xs.add((x,y))
            agents.append([x, y, self.max_hp, 0, 0, 0])  # x,y,hp,fx,fy,cd
        return np.array(agents, dtype=np.int32)

    def _line_of_fire_hits(self, shooter, enemies):
        sx, sy, _, fx, fy, _ = shooter
        if fx == 0 and fy == 0:
            return -1, None  # no facing
        if fx != 0: fx = 1 if fx > 0 else -1
        if fy != 0: fy = 1 if fy > 0 else -1

        candidates = []
        for idx, (ex, ey, ehp, *_rest) in enumerate(enemies):
            if ehp <= 0: continue
            dx = ex - sx; dy = ey - sy
            if fx == 0 and dx == 0 and dy != 0 and (np.sign(dy) == fy or fy == 0):
                candidates.append((abs(dy), idx, (sx, sy), (ex, ey)))
            elif fy == 0 and dy == 0 and dx != 0 and (np.sign(dx) == fx or fx == 0):
                candidates.append((abs(dx), idx, (sx, sy), (ex, ey)))
            elif abs(dx) == abs(dy) and dx != 0:
                sdx = 1 if dx > 0 else -1
                sdy = 1 if dy > 0 else -1
                if (sdx == fx or fx == 0) and (sdy == fy or fy == 0):
                    candidates.append((abs(dx), idx, (sx, sy), (ex, ey)))
        if not candidates:
            # 히트 실패: 방향으로 쭉 뻗는 맵 경계 교차점까지를 to로 만든다
            # (시각화용; 끝점은 화면 밖이 아닌 맵 경계)
            tx, ty = sx, sy
            while 0 <= tx < self.width and 0 <= ty < self.height:
                tx += fx; ty += fy
            # 경계 밖으로 한 칸 나갔으니 한 칸 되돌림
            tx -= fx; ty -= fy
            return -1, (sx, sy, tx, ty)
        candidates.sort(key=lambda t: t[0])
        _dist, idx, (sx, sy), (ex, ey) = candidates[0]
        return idx, (sx, sy, ex, ey)

    def _clamp_xy(self, x, y):
        return max(0, min(self.width-1, x)), max(0, min(self.height-1, y))

    def _alive_mask(self, team):
        return (team[:,2] > 0)

    def _apply_actions(self, team, acts):
        for i, a in enumerate(acts):
            if team[i,2] <= 0:  # dead
                continue
            if a == self.ACT_STAY or a == self.ACT_ATTACK:
                continue
            if a in self.ACT_MOVE_DIRS:
                dx, dy = self.ACT_MOVE_DIRS[a]
                nx, ny = self._clamp_xy(team[i,0] + dx, team[i,1] + dy)
                team[i,0], team[i,1] = nx, ny
                team[i,3], team[i,4] = dx, dy
        return (acts == self.ACT_ATTACK)

    def step(self, acts_A, acts_B):
        assert len(acts_A) == self.n and len(acts_B) == self.n
        time_penalty_A = -0.005 * self._alive_mask(self.A).sum()
        time_penalty_B = -0.005 * self._alive_mask(self.B).sum()

        self.A[:,5] = np.maximum(0, self.A[:,5] - 1)
        self.B[:,5] = np.maximum(0, self.B[:,5] - 1)

        shoot_A = self._apply_actions(self.A, acts_A)
        shoot_B = self._apply_actions(self.B, acts_B)

        rA = time_penalty_A; rB = time_penalty_B
        shots = []  # ★ 시각화용 발사 이벤트

        # A 발사
        for i, do_shoot in enumerate(shoot_A):
            if not do_shoot: continue
            if self.A[i,2] <= 0 or self.A[i,5] > 0: continue
            j, seg = self._line_of_fire_hits(self.A[i], self.B)
            if seg is not None:
                sx, sy, tx, ty = seg
            else:
                sx = self.A[i,0]; sy = self.A[i,1]; tx, ty = sx, sy
            if j >= 0:
                self.B[j,2] -= 1
                rA += 1.0; rB -= 1.0
                shots.append({"team":"A","i":i,"from":[sx,sy],"to":[int(self.B[j,0]),int(self.B[j,1])],"hit":True})
            else:
                shots.append({"team":"A","i":i,"from":[sx,sy],"to":[tx,ty],"hit":False})
            self.A[i,5] = self.reload_steps

        # B 발사
        for i, do_shoot in enumerate(shoot_B):
            if not do_shoot: continue
            if self.B[i,2] <= 0 or self.B[i,5] > 0: continue
            j, seg = self._line_of_fire_hits(self.B[i], self.A)
            if seg is not None:
                sx, sy, tx, ty = seg
            else:
                sx = self.B[i,0]; sy = self.B[i,1]; tx, ty = sx, sy
            if j >= 0:
                self.A[j,2] -= 1
                rB += 1.0; rA -= 1.0
                shots.append({"team":"B","i":i,"from":[sx,sy],"to":[int(self.A[j,0]),int(self.A[j,1])],"hit":True})
            else:
                shots.append({"team":"B","i":i,"from":[sx,sy],"to":[tx,ty],"hit":False})
            self.B[i,5] = self.reload_steps

        self.t += 1
        done = False
        if self._alive_mask(self.A).sum() == 0 and self._alive_mask(self.B).sum() == 0:
            done = True
        elif self._alive_mask(self.A).sum() == 0:
            done = True; rA -= 2.0; rB += 2.0
        elif self._alive_mask(self.B).sum() == 0:
            done = True; rA += 2.0; rB -= 2.0
        elif self.t >= self.max_steps:
            done = True

        info = {"shots": shots}  # ★ 추가
        return self._obs_team(self.A, self.B), self._obs_team(self.B, self.A), float(rA), float(rB), done, info

    def _obs_team(self, me, other):
        return np.concatenate([me.flatten(), other.flatten()]).astype(np.int32)

    def sample_actions(self):
        return np.random.randint(0, 10, size=self.n, dtype=np.int32)
