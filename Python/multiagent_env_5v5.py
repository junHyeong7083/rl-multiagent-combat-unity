# multiagent_env_5v5.py
import numpy as np
from dataclasses import dataclass

# ---- 액션 정의 ----
# 0: Stay
# 1: Up (0, +1)
# 2: Down (0, -1)
# 3: Left (-1, 0)
# 4: Right (+1, 0)
# 5: Shoot Up
# 6: Shoot Down
# 7: Shoot Left
# 8: Shoot Right
# 9: Shoot Nearest (직선 가시선 우선, 없으면 같은 행/열만 고려)
ACTION_NUM = 10


@dataclass
class Shot:
    from_x: int
    from_y: int
    to_x:   int
    to_y:   int
    team:   str   # "A" or "B"
    hit:    bool


class CombatSelfPlay5v5Env:
    """
    간단한 그리드 2팀 전투 환경 (A vs B).

    - 맵: width x height (정수 그리드)
    - 팀당 n개 유닛, 각 유닛은 [x,y,hp,fx,fy,cd]로 상태 저장
      (fx, fy = 마지막 이동/사격 방향 벡터 표시용, cd = 공격 쿨다운 틱)
    - 장애물: self.blocks[x, y] == True 이면 통과 불가 / 사격 차단
    - 관측(obs): 전역 관측(팀 공통) — 양 스크립트가 동일 obs_dim을 사용
      obs_dim = 6*n + 4  # (A: x,y,hp)*n + (B: x,y,hp)*n + (baseA x,y) + (baseB x,y)
      모든 위치는 [0,1]로 정규화, hp는 [0,1] 정규화
    - 보상: (가한 데미지 합) - (받은 데미지 합)
    """
    def __init__(self,
                 width: int = 24,
                 height: int = 24,
                 n_per_team: int = 6,
                 max_steps: int = 600,
                 seed: int = 0,
                 use_obstacles: bool = True,
                 obstacle_density: float = 0.08,
                 hp_max: int = 5,
                 shoot_range: int = 7,
                 move_cooldown: int = 0,
                 attack_cooldown: int = 20):
        self.width = int(width)
        self.height = int(height)
        self.n = int(n_per_team)
        self.max_steps = int(max_steps)
        self.rng = np.random.RandomState(seed)
        self.use_obstacles = bool(use_obstacles)
        self.obstacle_density = float(obstacle_density)
        self.hp_max = int(hp_max)
        self.shoot_range = int(shoot_range)
        self.move_cd = int(move_cooldown)
        self.attack_cd = int(attack_cooldown)

        # 뷰어/외부 호환 필드
        self.obs_ver = 1
        self.cell_size = 1.0

        # 상태
        self.t = 0
        self.base_A = np.array([1, self.height // 2], dtype=np.int32)
        self.base_B = np.array([self.width - 2, self.height // 2], dtype=np.int32)

        # 장애물 맵 (shape [W, H])
        self.blocks = np.zeros((self.width, self.height), dtype=bool)

        # 유닛 상태 배열 (A/B): shape [n, 6] = [x,y,hp,fx,fy,cd]
        self.A = np.zeros((self.n, 6), dtype=np.int32)
        self.B = np.zeros((self.n, 6), dtype=np.int32)

        self._place_obstacles()
        self.reset()

    # ====== 유틸 ======
    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_blocked(self, x: int, y: int) -> bool:
        # 경계 → 장애물 순서로 검사
        if not self._in_bounds(x, y):
            return True
        return self.use_obstacles and self.blocks[x, y]

    def _rand_empty_cell(self):
        # 장애물/기지 아닌 빈칸 임의 선택
        for _ in range(1000):
            x = self.rng.randint(0, self.width)
            y = self.rng.randint(0, self.height)
            if self.blocks[x, y]:
                continue
            if (x == self.base_A[0] and y == self.base_A[1]) or (x == self.base_B[0] and y == self.base_B[1]):
                continue
            return x, y
        # fallback
        return 0, 0

    # ====== 초기화 ======
    def _place_obstacles(self):
        self.blocks.fill(False)
        if not self.use_obstacles:
            return
        # 좌/우 기지 근처는 비워둠
        margin = 2
        for x in range(self.width):
            for y in range(self.height):
                if x < margin or x >= self.width - margin:
                    continue
                if self.rng.rand() < self.obstacle_density:
                    self.blocks[x, y] = True

        # 기지-기지 중앙 라인 통로 조금 뚫기
        cy = self.height // 2
        for x in range(self.base_A[0], self.base_B[0] + 1):
            self.blocks[x, cy] = False

    def _spawn_team_line(self, left: bool):
        # 왼쪽/오른쪽 라인에 유닛 배치 (중앙 근처)
        arr = np.zeros((self.n, 6), dtype=np.int32)
        col = 1 if left else (self.width - 2)
        ys = np.linspace(1, self.height - 2, self.n, dtype=int)
        for i, y in enumerate(ys):
            x = col
            # 빈칸 보정
            if self._is_blocked(x, y):
                x, y = self._rand_empty_cell()
            arr[i, 0] = x
            arr[i, 1] = y
            arr[i, 2] = self.hp_max   # hp
            arr[i, 3] = 0             # fx
            arr[i, 4] = 0             # fy
            arr[i, 5] = 0             # cd
        return arr

    def reset(self):
        self.t = 0
        # 기지 위치
        self.base_A = np.array([1, self.height // 2], dtype=np.int32)
        self.base_B = np.array([self.width - 2, self.height // 2], dtype=np.int32)
        # 유닛 배치
        self.A = self._spawn_team_line(left=True)
        self.B = self._spawn_team_line(left=False)
        return self._team_obs(), self._team_obs()

    # ====== 관측 ======
    def get_team_obs_dim(self) -> int:
        # (A: x,y,hp)*n + (B: x,y,hp)*n + (baseA x,y) + (baseB x,y)
        return 6 * self.n + 4

    def _norm_xy(self, x, y):
        return x / (self.width - 1.0), y / (self.height - 1.0)

    def _team_obs(self):
        # 전역 관측 (팀 공통)
        obs = []
        # A 팀 유닛
        for i in range(self.n):
            x, y, hp = self.A[i, 0], self.A[i, 1], self.A[i, 2]
            nx, ny = self._norm_xy(x, y)
            obs += [nx, ny, hp / self.hp_max]
        # B 팀 유닛
        for i in range(self.n):
            x, y, hp = self.B[i, 0], self.B[i, 1], self.B[i, 2]
            nx, ny = self._norm_xy(x, y)
            obs += [nx, ny, hp / self.hp_max]
        # 기지
        bax, bay = self._norm_xy(self.base_A[0], self.base_A[1])
        bbx, bby = self._norm_xy(self.base_B[0], self.base_B[1])
        obs += [bax, bay, bbx, bby]
        return np.asarray(obs, dtype=np.float32)

    # ====== 외부로 내보낼 장애물 정보 ======
    def export_obstacles(self):
        if not self.use_obstacles:
            return []
        coords = np.argwhere(self.blocks)  # [[x,y], ...]
        return [[int(x), int(y)] for x, y in coords]

    # ====== 스텝 ======
    def step(self, actionsA: np.ndarray, actionsB: np.ndarray):
        """
        actionsA / actionsB: shape [n] 의 정수 액션
        returns: obsA, obsB, rA, rB, done, info
        """
        self.t += 1
        shots = []

        # 쿨다운 감소
        self.A[:, 5] = np.maximum(0, self.A[:, 5] - 1)
        self.B[:, 5] = np.maximum(0, self.B[:, 5] - 1)

        # 이동 단계
        self._move_phase(self.A, actionsA)
        self._move_phase(self.B, actionsB)

        # 공격 단계 (적 존재 시에만 발사/쿨다운 소모, 아군은 차폐 아님)
        dmg_AtoB, shots_A = self._attack_phase(self.A, actionsA, self.B, "A")
        dmg_BtoA, shots_B = self._attack_phase(self.B, actionsB, self.A, "B")
        shots.extend(shots_A); shots.extend(shots_B)

        # 사망 처리 (hp <= 0 비활성화)
        self.A[:, 2] = np.maximum(0, self.A[:, 2])
        self.B[:, 2] = np.maximum(0, self.B[:, 2])

        # 보상: 가한 피해 - 받은 피해
        rA = float(dmg_AtoB - dmg_BtoA)
        rB = float(dmg_BtoA - dmg_AtoB)

        done = self._terminal()
        info = {
            "t": int(self.t),
            "shots": shots,
            "A_nexus": [int(self.base_A[0]), int(self.base_A[1])],
            "B_nexus": [int(self.base_B[0]), int(self.base_B[1])],
            "outcome": self._outcome() if done else None
        }
        obs = self._team_obs()
        return obs, obs, rA, rB, done, info

    def _terminal(self) -> bool:
        if self.t >= self.max_steps:
            return True
        aliveA = int((self.A[:, 2] > 0).sum())
        aliveB = int((self.B[:, 2] > 0).sum())
        if aliveA == 0 or aliveB == 0:
            return True
        return False

    def _outcome(self):
        aliveA = int((self.A[:, 2] > 0).sum())
        aliveB = int((self.B[:, 2] > 0).sum())
        if aliveA > aliveB:
            return "A"
        elif aliveB > aliveA:
            return "B"
        else:
            return "draw"

    # ====== 이동 단계 ======
    def _move_phase(self, team_arr: np.ndarray, actions: np.ndarray):
        # 0~4 (Stay/UDLR)만 이동 처리
        for i in range(self.n):
            if team_arr[i, 2] <= 0:
                continue  # 사망
            a = int(actions[i])
            if a < 0 or a > 9:
                continue
            dx, dy = 0, 0
            if a == 1:   dy = +1
            elif a == 2: dy = -1
            elif a == 3: dx = -1
            elif a == 4: dx = +1
            else:
                # 이동 아님
                continue

            nx = int(team_arr[i, 0]) + dx
            ny = int(team_arr[i, 1]) + dy

            # 경계 → 장애물 순서로 체크
            if not self._in_bounds(nx, ny):
                continue
            if self.use_obstacles and self.blocks[nx, ny]:
                continue

            team_arr[i, 0] = nx
            team_arr[i, 1] = ny
            team_arr[i, 3] = dx
            team_arr[i, 4] = dy

    # ====== 레이에서 '적만' 탐색 (아군은 차폐 X) ======
    def _first_enemy_on_ray(self, x0: int, y0: int, dx: int, dy: int, defenders: np.ndarray):
        """
        (x0,y0)에서 (dx,dy)로 shoot_range까지 전진.
        - 장애물/경계 만나면 중단
        - 아군은 무시(차폐 X)
        - 처음 만난 '적'을 반환
        return (found:bool, tx:int, ty:int, enemy_idx:int)
        """
        if dx == 0 and dy == 0:
            dx, dy = 1, 0

        cx, cy = x0, y0
        for _ in range(self.shoot_range):
            nx, ny = cx + dx, cy + dy

            # 1) 경계 체크
            if not self._in_bounds(nx, ny):
                return False, cx, cy, -1

            # 2) 장애물 체크 (막힘)
            if self.use_obstacles and self.blocks[nx, ny]:
                return False, cx, cy, -1

            # 3) 적 명중 검사 (아군은 완전히 무시)
            for j in range(self.n):
                if defenders[j, 2] > 0 and defenders[j, 0] == nx and defenders[j, 1] == ny:
                    return True, nx, ny, j

            cx, cy = nx, ny

        return False, cx, cy, -1

    def _try_fire(self, x0, y0, dx, dy, defenders, side: str):
        """
        적이 레이에 있을 때만 발사/적 체력-1/쿨다운 소비.
        없으면 아무 것도 안 함(쿨다운 미소비).
        """
        found, tx, ty, idx = self._first_enemy_on_ray(x0, y0, dx, dy, defenders)
        if found and idx >= 0:
            defenders[idx, 2] -= 1
            return 1, Shot(x0, y0, tx, ty, side, True), True
        return 0, None, False

    # ====== 공격 단계 ======
    def _attack_phase(self, attackers: np.ndarray, actions: np.ndarray,
                      defenders: np.ndarray, side: str):
        """
        아군은 차폐로 보지 않음. 장애물/경계만 차폐.
        레이에 '적'이 있을 때에만 발사하고, 그 때만 쿨다운을 소비.
        """
        total_damage = 0
        shots = []
        for i in range(self.n):
            if attackers[i, 2] <= 0:
                continue  # 사망
            a = int(actions[i])
            if a < 5 or a > 9:
                continue  # 사격 아님
            if attackers[i, 5] > 0:
                continue  # 쿨다운

            x0, y0 = int(attackers[i, 0]), int(attackers[i, 1])

            did_fire = False
            if a in (5, 6, 7, 8):
                # 명시적 방향
                if a == 5:   dx, dy = 0, +1
                elif a == 6: dx, dy = 0, -1
                elif a == 7: dx, dy = -1, 0
                else:        dx, dy = +1, 0
                dealt, shot, did_fire = self._try_fire(x0, y0, dx, dy, defenders, side)
                total_damage += dealt
                if shot: shots.append(shot)

            else:
                # 9: Nearest (같은 행/열 중 가장 가까운 적을 우선)
                best = None  # (dist, dx, dy)
                for j in range(self.n):
                    if defenders[j, 2] <= 0:
                        continue
                    ex, ey = int(defenders[j, 0]), int(defenders[j, 1])
                    if ex == x0:          # 수직
                        dy = 1 if ey > y0 else -1
                        dist = abs(ey - y0)
                        best = min(best, (dist, 0, dy)) if best else (dist, 0, dy)
                    elif ey == y0:        # 수평
                        dx = 1 if ex > x0 else -1
                        dist = abs(ex - x0)
                        best = min(best, (dist, dx, 0)) if best else (dist, dx, 0)

                if best:
                    _, dx, dy = best
                    dealt, shot, did_fire = self._try_fire(x0, y0, dx, dy, defenders, side)
                    total_damage += dealt
                    if shot: shots.append(shot)
                # 같은 행/열 적이 없으면 사격 스킵(쿨다운 X)

            # 실제 발사했을 때만 쿨다운 갱신
            if did_fire:
                attackers[i, 5] = self.attack_cd

        return total_damage, shots
