# multiagent_env_5v5.py
import math
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np

# ----------------------------
# Actions
# ----------------------------
ACTION_STAY   = 0
ACTION_UP     = 1
ACTION_DOWN   = 2
ACTION_LEFT   = 3
ACTION_RIGHT  = 4
ACTION_ATTACK = 5

ACTION_NUM = 6  # 외부에서 import 해서 씁니다.


# ----------------------------
# Shot record for visualization
# ----------------------------
@dataclass
class Shot:
    team: str            # "A" or "B"
    from_xy: Tuple[int,int]
    to_xy: Tuple[int,int]
    hit: bool


def _rng(seed: int):
    r = np.random.RandomState(seed)
    return r


class CombatSelfPlay5v5Env:
    """
    격자 맵 위에서 A/B 두 팀이 싸우는 간단한 셀프플레이 환경.
    - 유닛 상태: [x, y, hp, face_x, face_y, cd]
    - cd(쿨다운)는 0일 때만 ATTACK 가능. 사격 시 cd_max로 설정됨.
    - 사거리: manhattan 거리 기준 self.range
    - 장애물(blocks) 존재 (use_obstacles=True)
    - 넥서스(팀 스폰 근처 좌표) 존재: A_nexus, B_nexus
    - step() 반환: (obsA, obsB, rA, rB, done, info)
    """

    def __init__(self,
                 width: int = 24,
                 height: int = 24,
                 n_per_team: int = 10,
                 max_steps: int = 120,
                 seed: int = 0,
                 use_obstacles: bool = True,
                 obstacle_rate: float = 0.06,
                 capture_radius: float = 1.5,
                 cd_max: int = 15):  # ★ 공격 쿨다운 기본 15틱
        self.width = int(width)
        self.height = int(height)
        self.n = int(n_per_team)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.rng = _rng(self.seed)
        self.use_obstacles = bool(use_obstacles)
        self.obstacle_rate = float(obstacle_rate)
        self.capture_radius = float(capture_radius)

        # combat 파라미터
        self.range = 5           # manhattan 사거리
        self.hp_max = 3          # 히트당 HP 1 감소, 0 이하 사망
        self.cd_max = int(cd_max)

        # 내부 상태
        self.t = 0
        self.blocks = None   # 2D boolean grid
        self.A = None        # [n,6]: x,y,hp,fx,fy,cd
        self.B = None
        self.shots: List[Shot] = []

        # 넥서스(스폰 근처): 좌표 지정
        # 맵 내부에서 벽과 너무 붙지 않게 살짝 띄워서 배치
        self.A_nexus = (1, 1)
        self.B_nexus = (self.width - 2, self.height - 2)

    # ----------------------------
    # Public API
    # ----------------------------
    def get_team_obs_dim(self) -> int:
        """
        팀 관측의 차원: (자기팀 유닛 n개 * 6) + (적팀 유닛 n개 * 6) + (양 팀 넥서스 4) + (time 1) = 12n + 5
        """
        return 12 * self.n + 5

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        환경 초기화. 두 팀의 팀 관측(obsA, obsB) 반환.
        """
        self.t = 0
        self.shots = []

        # 장애물 맵 생성
        if self.use_obstacles:
            self.blocks = (self.rng.rand(self.width, self.height) < self.obstacle_rate)
            # 넥서스 부근은 항상 비워둔다
            self._clear_radius(self.A_nexus, 2)
            self._clear_radius(self.B_nexus, 2)
        else:
            self.blocks = np.zeros((self.width, self.height), dtype=bool)

        # 유닛 초기화
        self.A = self._spawn_team_near(self.A_nexus)
        self.B = self._spawn_team_near(self.B_nexus)

        return self._build_team_obs_A(), self._build_team_obs_B()

    def step(self, aA: np.ndarray, aB: np.ndarray):
        """
        aA, aB: 길이 n의 액션 배열 (정수)
        반환: (obsA, obsB, rA, rB, done, info)
        """
        self.t += 1
        self.shots = []

        # 이동
        self._move(self.A, aA)
        self._move(self.B, aB)

        # 공격 (ATTACK 액션 + 쿨다운 0)인 유닛만 사격
        hitA = self._attack_phase(self.A, aA, self.B, "A")
        hitB = self._attack_phase(self.B, aB, self.A, "B")

        # 사망 처리(HP <= 0 -> 고정된 죽은 상태 유지)
        deadA = (self.A[:, 2] <= 0).sum()
        deadB = (self.B[:, 2] <= 0).sum()

        # 보상(간단히): 타격 +1, 사망 -1(비활성), 넥서스 근접 shaping 약간
        rA = float(hitA) + self._shaping_toward(self.A, self.B_nexus)
        rB = float(hitB) + self._shaping_toward(self.B, self.A_nexus)

        # 종료 조건: 캡처/시간 종료/전멸
        done, winA, winB = self._check_done()

        info = {
            "t": self.t,
            "hitsA": int(hitA),
            "hitsB": int(hitB),
            "deadA": int(deadA),
            "deadB": int(deadB),
            "winA": winA,
            "winB": winB,
            # 렌더링 보조
            "shots": self.shots,
            "blocks": self.blocks,
            "A_nexus": self.A_nexus,
            "B_nexus": self.B_nexus,
        }

        obsA = self._build_team_obs_A()
        obsB = self._build_team_obs_B()
        return obsA, obsB, rA, rB, bool(done), info

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _clear_radius(self, center: Tuple[int,int], r: int):
        cx, cy = center
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                x = np.clip(cx + dx, 0, self.width - 1)
                y = np.clip(cy + dy, 0, self.height - 1)
                self.blocks[x, y] = False

    def _spawn_team_near(self, base_xy: Tuple[int,int]) -> np.ndarray:
        """
        base_xy 근처에 n개 유닛을 배치. 겹침/장애물 칸 피해서 난수 배치.
        유닛 상태: [x, y, hp, fx, fy, cd]
        """
        units = np.zeros((self.n, 6), dtype=np.float32)
        bx, by = base_xy
        placed = 0
        tries = 0
        while placed < self.n and tries < 10000:
            tries += 1
            # 네 방향 +-3 타일 범위 내 랜덤
            rx = int(np.clip(bx + self.rng.randint(-3, 4), 0, self.width - 1))
            ry = int(np.clip(by + self.rng.randint(-3, 4), 0, self.height - 1))
            if self.blocks[rx, ry]:
                continue
            # 이미 같은 칸에 유닛 있는지 체크
            if self._occupied(rx, ry, units[:placed]):
                continue
            units[placed, 0] = rx
            units[placed, 1] = ry
            units[placed, 2] = self.hp_max
            units[placed, 3] = 0   # face_x
            units[placed, 4] = 1   # face_y
            units[placed, 5] = 0   # cd
            placed += 1

        # 혹시 다 못 놓았으면 빈칸에라도 채우기
        while placed < self.n:
            rx = int(np.clip(bx, 0, self.width - 1))
            ry = int(np.clip(by, 0, self.height - 1))
            if not self.blocks[rx, ry] and not self._occupied(rx, ry, units[:placed]):
                units[placed, 0] = rx
                units[placed, 1] = ry
                units[placed, 2] = self.hp_max
                units[placed, 3] = 0
                units[placed, 4] = 1
                units[placed, 5] = 0
                placed += 1
            else:
                bx = (bx + 1) % self.width
                by = (by + 1) % self.height

        return units

    def _occupied(self, x: int, y: int, arr_xyhp: np.ndarray) -> bool:
        for i in range(arr_xyhp.shape[0]):
            if (int(arr_xyhp[i, 0]) == x) and (int(arr_xyhp[i, 1]) == y) and (arr_xyhp[i,2] > 0):
                return True
        return False

    def _move(self, Team: np.ndarray, actions: np.ndarray):
        """
        이동 처리: 죽은 유닛은 무시. 장애물/경계/충돌 체크.
        """
        for i in range(self.n):
            if Team[i, 2] <= 0:
                continue  # dead

            ax = int(Team[i, 0])
            ay = int(Team[i, 1])

            a = int(actions[i])
            nx, ny = ax, ay
            fx, fy = Team[i, 3], Team[i, 4]

            if a == ACTION_UP:
                ny = ay - 1; fx, fy = 0, -1
            elif a == ACTION_DOWN:
                ny = ay + 1; fx, fy = 0, 1
            elif a == ACTION_LEFT:
                nx = ax - 1; fx, fy = -1, 0
            elif a == ACTION_RIGHT:
                nx = ax + 1; fx, fy = 1, 0
            elif a == ACTION_STAY or a == ACTION_ATTACK:
                # 제자리 / 공격(이동은 안 함)
                pass

            # 경계
            nx = int(np.clip(nx, 0, self.width - 1))
            ny = int(np.clip(ny, 0, self.height - 1))

            # 장애물/충돌 체크
            if not self.blocks[nx, ny] and not self._occupied(nx, ny, Team):
                Team[i, 0] = nx
                Team[i, 1] = ny
                Team[i, 3] = fx
                Team[i, 4] = fy

            # 쿨다운은 이동단계에서 깎지 않음 (공격 단계에서 일괄 감소)

    def _attack_phase(self, Att: np.ndarray, actions: np.ndarray, Def: np.ndarray, team_tag: str) -> int:
        """
        공격 단계: 각 유닛 쿨다운 1씩 감소 → ATTACK 액션 & cd==0인 유닛만 사격.
        맞추면 Def HP-1, Miss도 샷 기록만 남김.
        """
        # 1) 모든 어태커의 쿨다운 1틱 감소
        for i in range(self.n):
            if Att[i, 2] <= 0:
                continue
            if Att[i, 5] > 0:
                Att[i, 5] -= 1

        # 2) ATTACK + cd==0 인 유닛만 사격
        hits = 0
        for i in range(self.n):
            if Att[i, 2] <= 0:
                continue

            if int(actions[i]) != ACTION_ATTACK:
                continue
            if Att[i, 5] > 0:
                continue  # 아직 쿨다운

            ax, ay = int(Att[i, 0]), int(Att[i, 1])

            # 타겟(가장 가까운 적)
            alive_mask = (Def[:, 2] > 0)
            if not alive_mask.any():
                continue
            dxy = Def[:, :2] - Att[i, :2]
            dist = np.abs(dxy[:, 0]) + np.abs(dxy[:, 1])
            dist[~alive_mask] = 1e9
            tgt = int(np.argmin(dist))
            if not alive_mask[tgt]:
                continue

            if dist[tgt] <= self.range:
                # 적 피격: HP 1 감소
                Def[tgt, 2] -= 1
                hits += 1
                self.shots.append(
                    Shot(team=team_tag, from_xy=(ax, ay),
                         to_xy=(int(Def[tgt, 0]), int(Def[tgt, 1])), hit=True)
                )
            else:
                # miss: 연출용
                tx = int(Att[i, 0] + np.sign(dxy[tgt, 0]) * min(dist[tgt], self.range))
                ty = int(Att[i, 1] + np.sign(dxy[tgt, 1]) * min(dist[tgt], self.range))
                self.shots.append(
                    Shot(team=team_tag, from_xy=(ax, ay), to_xy=(tx, ty), hit=False)
                )

            # 사격 후 쿨다운 부여
            Att[i, 5] = self.cd_max

        return hits

    def _shaping_toward(self, Team: np.ndarray, goal_xy: Tuple[int,int]) -> float:
        """
        넥서스 방향 shaping 보상(아주 약하게).
        생존 중 유닛들의 평균 맨해튼 거리 감소를 보상으로.
        """
        gx, gy = goal_xy
        alive = (Team[:, 2] > 0)
        if not alive.any():
            return 0.0
        d = np.abs(Team[alive, 0] - gx) + np.abs(Team[alive, 1] - gy)
        # 거리가 작을 수록 + (최대 맵 크기로 정규화)
        max_d = self.width + self.height
        val = float((max_d - d.mean()) / max_d) * 0.02  # 매우 약하게
        return val

    def _check_done(self):
        """
        종료/승패 판정:
        - 어느 팀이든 넥서스 반경 capture_radius 내로 다다르면 그 팀 승리
        - 전멸 시 상대 승리
        - 스텝 초과 시 무승부
        """
        # 캡처 판정
        if self._any_in_radius(self.A, self.B_nexus, self.capture_radius):
            return True, True, False
        if self._any_in_radius(self.B, self.A_nexus, self.capture_radius):
            return True, False, True

        # 전멸 판정
        aliveA = (self.A[:, 2] > 0).sum()
        aliveB = (self.B[:, 2] > 0).sum()
        if aliveA == 0 and aliveB > 0:
            return True, False, True
        if aliveB == 0 and aliveA > 0:
            return True, True, False
        if aliveA == 0 and aliveB == 0:
            return True, False, False

        # 스텝 초과
        if self.t >= self.max_steps:
            return True, False, False

        return False, False, False

    def _any_in_radius(self, Team: np.ndarray, pt: Tuple[int,int], radius: float) -> bool:
        if (Team[:, 2] > 0).sum() == 0:
            return False
        gx, gy = pt
        dx = Team[:, 0] - gx
        dy = Team[:, 1] - gy
        dist = np.sqrt(dx * dx + dy * dy)
        return bool((dist <= radius).any())

    # ----------------------------
    # Observations (Team-level)
    # ----------------------------
    def _build_team_obs_A(self) -> np.ndarray:
        return self._build_team_obs(self.A, self.B, self.A_nexus, self.B_nexus)

    def _build_team_obs_B(self) -> np.ndarray:
        return self._build_team_obs(self.B, self.A, self.B_nexus, self.A_nexus)

    def _build_team_obs(self,
                        Own: np.ndarray,
                        Opp: np.ndarray,
                        own_nexus: Tuple[int,int],
                        opp_nexus: Tuple[int,int]) -> np.ndarray:
        """
        팀 관측: [Own(n*6), Opp(n*6), own_nexus(2), opp_nexus(2), t_norm(1)]
        """
        own_flat = Own.reshape(-1)                     # 6n
        opp_flat = Opp.reshape(-1)                     # 6n
        t_norm = np.array([self.t / max(1, self.max_steps)], dtype=np.float32)
        obs = np.concatenate([
            own_flat, opp_flat,
            np.array(own_nexus, dtype=np.float32),
            np.array(opp_nexus, dtype=np.float32),
            t_norm
        ]).astype(np.float32)
        return obs

    # ----------------------------
    # Streaming frame (for LiveViewer5v5 UDP)
    # ----------------------------
    def make_frame(self):
        """
        LiveViewer5v5(UDP)에서 기대하는 프레임 포맷:
        {t,width,height,base_A,base_B,A,B,shots,outcome}
        - A/B: [[x,y,hp,fx,fy,cd], ...] 길이 n
        - base_A/base_B: [x:int, y:int]
        - shots: [{team:"A"|"B", from:[x,y], to:[x,y], hit:bool}, ...]
        - outcome: "A" | "B" | "draw" | None
        """
        def to_list_int(arr):
            # x,y,hp,fx,fy,cd 순서 유지
            a = arr[:, :6].copy()
            a[:, 0] = np.clip(a[:, 0], 0, self.width - 1)
            a[:, 1] = np.clip(a[:, 1], 0, self.height - 1)
            return [[int(v0), int(v1), int(v2), int(v3), int(v4), int(v5)]
                    for (v0, v1, v2, v3, v4, v5) in a]

        A_list = to_list_int(self.A)
        B_list = to_list_int(self.B)

        # 샷 기록 변환
        shots_list = []
        for s in self.shots:
            shots_list.append({
                "team": s.team,
                "from": [int(s.from_xy[0]), int(s.from_xy[1])],
                "to":   [int(s.to_xy[0]),   int(s.to_xy[1])],
                "hit":  bool(s.hit),
            })

        # 결과 표시(아직 진행 중이면 None)
        done, winA, winB = self._check_done()
        if not done:
            outcome = None
        else:
            if winA and not winB:
                outcome = "A"
            elif winB and not winA:
                outcome = "B"
            else:
                outcome = "draw"

        frame = {
            "t": int(self.t),
            "width": int(self.width),
            "height": int(self.height),
            # ★ 키 이름을 base_A / base_B 로 통일 (Unity 뷰어가 이 키를 읽음)
            "base_A": [int(self.A_nexus[0]), int(self.A_nexus[1])],
            "base_B": [int(self.B_nexus[0]), int(self.B_nexus[1])],
            "A": A_list,
            "B": B_list,
            "shots": shots_list,
            "outcome": outcome,
        }
        return frame
