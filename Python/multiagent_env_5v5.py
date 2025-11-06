import numpy as np

class CombatSelfPlay5v5Env:
    """
    5v5 격자 전투 환경 (동시행동, 자체 심플 LOS)
    - 8방향 이동 가능(대각 포함), 가만히 있기
    - 바라보는 방향(facing)으로 '사거리 무한 직선' 발사 (벽 없음 가정)
    - 재장전 쿨다운: 5 스텝
    - 한 발에 HP 1 감소, 기본 HP=3
    - 아군/적군이 모두 쓰러지거나 max_steps 도달 시 종료
    - 팀 보상: (명중 +1, 피격 -1), 소폭 시간 패널티(-0.005/에이전트)
    """
    # 액션: 0=대기, 1~8=이동(상,하,좌,우,좌상,우상,좌하,우하), 9=공격
    ACT_STAY = 0
    ACT_MOVE_DIRS = {
        1: (0, -1),  # up
        2: (0,  1),  # down
        3: (-1, 0),  # left
        4: (1,  0),  # right
        5: (-1,-1),  # up-left
        6: (1, -1),  # up-right
        7: (-1, 1),  # down-left
        8: (1,  1),  # down-right
    }
    ACT_ATTACK = 9

    def __init__(self, width=12, height=8, n_per_team=5, max_steps=100,
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
        # A는 좌측, B는 우측에 랜덤 스폰(겹치지 않도록)
        self.A = self._spawn_side(left=True)
        self.B = self._spawn_side(left=False)
        return self._obs_team(self.A, self.B), self._obs_team(self.B, self.A)

    # 에이전트 배열 구조: [ [x,y, hp, fx, fy, cd], ... ] 길이 n
    #  - (fx, fy)는 바라보는 방향의 정규화된 벡터(0,0) 허용 (최근 이동 방향 유지)
    #  - cd(cooldown): 남은 쿨다운 스텝(정수)
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
            agents.append([x, y, self.max_hp, 0, 0, 0])
        return np.array(agents, dtype=np.int32)

    def _line_of_fire_hits(self, shooter, enemies):
        """
        간단한 LOS: shooter의 (fx,fy) 방향 직선 상에 위치한 첫 번째 적을 맞춘다.
        (fx,fy)=(0,0)이면 사격 실패.
        같은 줄/대각선 정렬 여부로 판정.
        """
        sx, sy, _, fx, fy, _ = shooter
        if fx == 0 and fy == 0:
            return -1  # no facing
        # 정규화
        if fx != 0: fx = 1 if fx > 0 else -1
        if fy != 0: fy = 1 if fy > 0 else -1

        candidates = []
        for idx, (ex, ey, ehp, *_rest) in enumerate(enemies):
            if ehp <= 0:
                continue
            dx = ex - sx
            dy = ey - sy
            # 수직선
            if fx == 0 and dx == 0 and dy != 0 and (np.sign(dy) == fy or fy == 0):
                candidates.append((abs(dy), idx))
            # 수평선
            elif fy == 0 and dy == 0 and dx != 0 and (np.sign(dx) == fx or fx == 0):
                candidates.append((abs(dx), idx))
            # 대각선(기울기 1)
            elif abs(dx) == abs(dy) and dx != 0:
                sdx = 1 if dx > 0 else -1
                sdy = 1 if dy > 0 else -1
                if (sdx == fx or fx == 0) and (sdy == fy or fy == 0):
                    candidates.append((abs(dx), idx))

        if not candidates:
            return -1
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def _clamp_xy(self, x, y):
        return max(0, min(self.width-1, x)), max(0, min(self.height-1, y))

    def _alive_mask(self, team):
        return (team[:,2] > 0)

    def _apply_actions(self, team, acts):
        """
        이동/공격 처리. 이동 먼저, 그 다음 일괄 공격 처리.
        이동 시 겹침 허용(단순화). 이동한 방향이 있으면 facing 갱신.
        공격은 cd==0 이고 hp>0 일 때만 가능. 공격 시 cd=reload_steps.
        """
        # 이동 처리
        for i, a in enumerate(acts):
            if team[i,2] <= 0:  # dead
                continue
            if a == self.ACT_STAY or a == self.ACT_ATTACK:
                continue
            if a in self.ACT_MOVE_DIRS:
                dx, dy = self.ACT_MOVE_DIRS[a]
                nx, ny = self._clamp_xy(team[i,0] + dx, team[i,1] + dy)
                team[i,0], team[i,1] = nx, ny
                # facing 갱신
                team[i,3], team[i,4] = dx, dy

        # 공격 요청 수집 (사격은 일괄 판정)
        shoot_flags = (acts == self.ACT_ATTACK)
        return shoot_flags

    def step(self, acts_A, acts_B):
        """
        acts_* : 길이 n ndarray[int], 각 에이전트의 액션
        return:
            obs_A, obs_B, rA, rB, done, info
        """
        assert len(acts_A) == self.n and len(acts_B) == self.n

        # 기본 시간 패널티(모든 생존자에 대해)
        time_penalty_A = -0.005 * self._alive_mask(self.A).sum()
        time_penalty_B = -0.005 * self._alive_mask(self.B).sum()

        # 쿨다운 감소
        self.A[:,5] = np.maximum(0, self.A[:,5] - 1)
        self.B[:,5] = np.maximum(0, self.B[:,5] - 1)

        # 이동
        shoot_A = self._apply_actions(self.A, acts_A)
        shoot_B = self._apply_actions(self.B, acts_B)

        # 공격 판정
        rA = time_penalty_A
        rB = time_penalty_B

        # A가 쏘기
        for i, do_shoot in enumerate(shoot_A):
            if not do_shoot: 
                continue
            if self.A[i,2] <= 0 or self.A[i,5] > 0:
                continue
            j = self._line_of_fire_hits(self.A[i], self.B)
            if j >= 0:
                self.B[j,2] -= 1
                rA += 1.0
                rB -= 1.0
            # 쿨다운 갱신
            self.A[i,5] = self.reload_steps

        # B가 쏘기
        for i, do_shoot in enumerate(shoot_B):
            if not do_shoot: 
                continue
            if self.B[i,2] <= 0 or self.B[i,5] > 0:
                continue
            j = self._line_of_fire_hits(self.B[i], self.A)
            if j >= 0:
                self.A[j,2] -= 1
                rB += 1.0
                rA -= 1.0
            self.B[i,5] = self.reload_steps

        self.t += 1

        done = False
        if self._alive_mask(self.A).sum() == 0 and self._alive_mask(self.B).sum() == 0:
            done = True
        elif self._alive_mask(self.A).sum() == 0:
            done = True
            rA -= 2.0
            rB += 2.0
        elif self._alive_mask(self.B).sum() == 0:
            done = True
            rA += 2.0
            rB -= 2.0
        elif self.t >= self.max_steps:
            done = True

        return self._obs_team(self.A, self.B), self._obs_team(self.B, self.A), float(rA), float(rB), done, {}

    def _obs_team(self, me, other):
        """
        관측: 팀 전체 단일 벡터(간단 중앙집중 관측)
        각 에이전트당 6개(x,y,hp,fx,fy,cd) => n*6*2 (아군 + 적군)
        [A_agents_flat, B_agents_flat] (정수형)
        """
        obs = np.concatenate([me.flatten(), other.flatten()]).astype(np.int32)
        return obs

    # 액션 샘플링 헬퍼
    def sample_actions(self):
        return np.random.randint(0, 10, size=self.n, dtype=np.int32)
