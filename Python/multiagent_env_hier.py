import numpy as np

# ====== 맵/유닛 파라미터(필요에 맞게 조절) ======
MAP_W = 30
MAP_H = 30
MAX_STEPS = 300

N_PER_TEAM = 5
N_ACTIONS  = 6  # 0 stay, 1 up, 2 down, 3 left, 4 right, 5 shoot

# ====== 보상 상수 (스케일 크게 / 방향성 소수점) ======
REWARD_KILL          = +4.0
REWARD_DEATH         = -4.0
REWARD_HIT           = +0.5
REWARD_CAPTURE       = +20.0
REWARD_TIMEOUT_WIN   = +6.0
REWARD_TIMEOUT_LOSS  = -6.0

STEP_TOWARD_BASE     = +0.08
STEP_AWAY_FROM_BASE  = -0.04
STALL_PENALTY        = -0.02

# 타임아웃 점수 가중
ALIVE_WEIGHT     = 1.0
PROGRESS_WEIGHT  = 0.03

# 캡처 판정
CAPTURE_RADIUS   = 1.5

class CombatSelfPlayHierEnv:
    """
    간단화된 2D 격자 환경(자기대전). 
    관측(obs_dim)= 11 (예시), 커맨더 1-hot은 모델 쪽에서 붙임.
    """
    def __init__(self, n_per_team=N_PER_TEAM, seed=123):
        self.rng = np.random.RandomState(seed)
        self.n = n_per_team
        self.n_actions = N_ACTIONS
        self.obs_dim = 11  # 기본 특성 개수 (유닛 위치/벡터/체력/거리 등 구성했다고 가정)
        self.MAX_STEPS = MAX_STEPS
        self.reset()

    # ---------------- 유틸 ----------------
    def _rand_pos_in_box(self, x0, y0, x1, y1, k):
        xs = self.rng.randint(x0, x1+1, size=k)
        ys = self.rng.randint(y0, y1+1, size=k)
        return np.stack([xs, ys], axis=1).astype(np.float32)

    def _min_dist_to_enemy_base(self, pos_alive, nexus_xy):
        if pos_alive.shape[0] == 0:
            return 1e9
        d = np.linalg.norm(pos_alive - nexus_xy[None, :], axis=1)
        return float(np.min(d))

    def _team_timeout_score(self, side: str) -> float:
        if side == 'A':
            alive = int((self.hpA > 0).sum())
            cur_min = self._min_dist_to_enemy_base(self.posA[self.hpA > 0], self.nexusB)
            prog = max(0.0, float(self._start_minA_to_B - cur_min))
        else:
            alive = int((self.hpB > 0).sum())
            cur_min = self._min_dist_to_enemy_base(self.posB[self.hpB > 0], self.nexusA)
            prog = max(0.0, float(self._start_minB_to_A - cur_min))
        return ALIVE_WEIGHT * float(alive) + PROGRESS_WEIGHT * prog

    def _observe_team(self, team: str):
        """
        간단화를 위해 유닛별로 동일한 팀 전역 관측을 돌려준다(형상: (N, obs_dim)).
        여기서는 예시로 [내 평균x, 평균y, 내 넥서스x,y, 적 넥서스x,y, 최소거리, 남은아군, 남은적군, step_norm, map_w, map_h]
        """
        if team == 'A':
            my_pos = self.posA; my_hp = self.hpA
            nx, ny = self.nexusB
            ex_alive = int((self.hpB > 0).sum())
            me_alive = int((self.hpA > 0).sum())
            min_d = self._min_dist_to_enemy_base(my_pos[my_hp>0], self.nexusB)
        else:
            my_pos = self.posB; my_hp = self.hpB
            nx, ny = self.nexusA
            ex_alive = int((self.hpA > 0).sum())
            me_alive = int((self.hpB > 0).sum())
            min_d = self._min_dist_to_enemy_base(my_pos[my_hp>0], self.nexusA)

        if my_pos.shape[0] > 0:
            meanx, meany = float(my_pos[:,0].mean()), float(my_pos[:,1].mean())
        else:
            meanx, meany = 0.0, 0.0
        step_norm = self._step_count / float(self.MAX_STEPS)

        base = np.array([meanx, meany, 
                         nx, ny, 
                         self.nexusA[0], self.nexusA[1],
                         min_d, me_alive, ex_alive, 
                         step_norm, MAP_W], dtype=np.float32)
        # obs_dim==11로 맞추기
        # (마지막에 MAP_H 넣고 싶다면 obs_dim=12로 맞춰 모델/훈련 코드도 바꿔야 함)
        out = np.tile(base[None, :], (self.n, 1))
        return out

    # ---------------- API ----------------
    def reset(self):
        # 시작 위치: A=좌하(7시), B=우상(1시)
        self.nexusA = np.array([2.0, 2.0], dtype=np.float32)
        self.nexusB = np.array([MAP_W-3.0, MAP_H-3.0], dtype=np.float32)

        self.posA = self._rand_pos_in_box(1, 1, 5, 5, self.n)
        self.posB = self._rand_pos_in_box(MAP_W-6, MAP_H-6, MAP_W-2, MAP_H-2, self.n)

        self.hpA = np.ones((self.n,), dtype=np.int32) * 5
        self.hpB = np.ones((self.n,), dtype=np.int32) * 5

        self._step_count = 0

        self._start_minA_to_B = self._min_dist_to_enemy_base(self.posA[self.hpA>0], self.nexusB)
        self._start_minB_to_A = self._min_dist_to_enemy_base(self.posB[self.hpB>0], self.nexusA)

        # 총알/샷 기록(유니티 시각화용)
        self.bullets = []      # list of dict {x,y,vx,vy,life,team}
        self.last_shots = []   # list of dict {team, fx,fy, tx,ty}

        obsA = self._observe_team('A')
        obsB = self._observe_team('B')
        return obsA, obsB

    def step(self, aA: np.ndarray, aB: np.ndarray):
        # --- 입력 안전 ---
        aA = aA.astype(np.int64)
        aB = aB.astype(np.int64)

        # --- 이전 최소거리 기록 (전진/후퇴 보상용) ---
        prev_minA = self._min_dist_to_enemy_base(self.posA[self.hpA>0], self.nexusB)
        prev_minB = self._min_dist_to_enemy_base(self.posB[self.hpB>0], self.nexusA)

        # --- 이동/사격 처리 ---
        self.last_shots = []

        def _apply_actions(pos, hp, act, team_char):
            # 이동
            for i in range(self.n):
                if hp[i] <= 0: 
                    continue
                ai = int(act[i])
                if ai == 1:   # up
                    pos[i,1] = min(MAP_H-1, pos[i,1] + 1.0)
                elif ai == 2: # down
                    pos[i,1] = max(0.0, pos[i,1] - 1.0)
                elif ai == 3: # left
                    pos[i,0] = max(0.0, pos[i,0] - 1.0)
                elif ai == 4: # right
                    pos[i,0] = min(MAP_W-1, pos[i,0] + 1.0)
                elif ai == 5: # shoot
                    # 간단히: 상대 넥서스 방향으로 발사
                    if team_char == 'A':
                        tx, ty = self.nexusB
                    else:
                        tx, ty = self.nexusA
                    fx, fy = pos[i,0], pos[i,1]
                    vx, vy = (tx - fx), (ty - fy)
                    n = np.linalg.norm([vx, vy]) + 1e-6
                    vx /= n; vy /= n
                    self.bullets.append({"x": fx, "y": fy, "vx": vx*2.0, "vy": vy*2.0, "life": 8, "team": team_char})
                    self.last_shots.append({"team": team_char, "fx": fx, "fy": fy, "tx": tx, "ty": ty})

        _apply_actions(self.posA, self.hpA, aA, 'A')
        _apply_actions(self.posB, self.hpB, aB, 'B')

        # --- 총알 이동 & 피격 판정 ---
        rA = np.zeros((self.n,), dtype=np.float32)
        rB = np.zeros((self.n,), dtype=np.float32)

        new_bullets = []
        for b in self.bullets:
            if b["life"] <= 0: 
                continue
            b["x"] += b["vx"]
            b["y"] += b["vy"]
            b["life"] -= 1
            if not (0 <= b["x"] < MAP_W and 0 <= b["y"] < MAP_H):
                continue

            # 피격: 반대팀만
            if b["team"] == 'A':
                tgt_pos, tgt_hp, rew_hit, rew_kill = self.posB, self.hpB, REWARD_HIT, REWARD_KILL
                my_rew_arr = rA
                opp_char = 'B'
            else:
                tgt_pos, tgt_hp, rew_hit, rew_kill = self.posA, self.hpA, REWARD_HIT, REWARD_KILL
                my_rew_arr = rB
                opp_char = 'A'

            # 단순 근접 판정
            hit_idx = None
            for i in range(self.n):
                if tgt_hp[i] <= 0: 
                    continue
                d = np.hypot(tgt_pos[i,0]-b["x"], tgt_pos[i,1]-b["y"])
                if d < 0.7:
                    hit_idx = i
                    break
            if hit_idx is not None:
                # 히트
                my_rew_arr += rew_hit / self.n  # 팀 분배(원하면 조정)
                tgt_hp[hit_idx] -= 1
                if tgt_hp[hit_idx] <= 0:
                    # 킬
                    my_rew_arr += rew_kill / self.n
                    if opp_char == 'A':
                        rA += REWARD_DEATH / self.n
                    else:
                        rB += REWARD_DEATH / self.n
                continue

            new_bullets.append(b)

        self.bullets = new_bullets

        # --- 전진/후퇴/교착 보상(팀 합산으로 적용) ---
        cur_minA = self._min_dist_to_enemy_base(self.posA[self.hpA>0], self.nexusB)
        cur_minB = self._min_dist_to_enemy_base(self.posB[self.hpB>0], self.nexusA)

        progA = prev_minA - cur_minA
        progB = prev_minB - cur_minB

        if progA > 0: rA += STEP_TOWARD_BASE
        elif progA < 0: rA += STEP_AWAY_FROM_BASE

        if progB > 0: rB += STEP_TOWARD_BASE
        elif progB < 0: rB += STEP_AWAY_FROM_BASE

        if abs(progA) < 1e-3 and abs(progB) < 1e-3:
            rA += STALL_PENALTY
            rB += STALL_PENALTY

        # --- 캡처 즉시 종료 체크 ---
        def _captured(pos, hp, nexus):
            if (hp > 0).any():
                d = np.min(np.linalg.norm(pos[hp > 0] - nexus[None, :], axis=1))
                return d <= CAPTURE_RADIUS
            return False

        done = False
        winner = None

        if _captured(self.posA, self.hpA, self.nexusB):
            done = True; winner = 'A'; rA += REWARD_CAPTURE; rB -= REWARD_CAPTURE/2
        elif _captured(self.posB, self.hpB, self.nexusA):
            done = True; winner = 'B'; rB += REWARD_CAPTURE; rA -= REWARD_CAPTURE/2

        # --- 전멸 ---
        if not done:
            aliveA = int((self.hpA > 0).sum())
            aliveB = int((self.hpB > 0).sum())
            if aliveA == 0 and aliveB == 0:
                done = True
                scoreA = self._team_timeout_score('A')
                scoreB = self._team_timeout_score('B')
                winner = 'A' if scoreA >= scoreB else 'B'
            elif aliveA == 0:
                done = True; winner = 'B'
            elif aliveB == 0:
                done = True; winner = 'A'

        # --- 타임아웃 ---
        self._step_count += 1
        if not done and self._step_count >= self.MAX_STEPS:
            scoreA = self._team_timeout_score('A')
            scoreB = self._team_timeout_score('B')
            done = True
            if scoreA > scoreB:
                winner = 'A'; rA += REWARD_TIMEOUT_WIN; rB += REWARD_TIMEOUT_LOSS
            elif scoreB > scoreA:
                winner = 'B'; rB += REWARD_TIMEOUT_WIN; rA += REWARD_TIMEOUT_LOSS
            else:
                # 동점 타이브레이커: 현재 최소거리
                curA = self._min_dist_to_enemy_base(self.posA[self.hpA>0], self.nexusB)
                curB = self._min_dist_to_enemy_base(self.posB[self.hpB>0], self.nexusA)
                if curA <= curB:
                    winner = 'A'; rA += REWARD_TIMEOUT_WIN; rB += REWARD_TIMEOUT_LOSS
                else:
                    winner = 'B'; rB += REWARD_TIMEOUT_WIN; rA += REWARD_TIMEOUT_LOSS

        # ---- 관측/리턴 ----
        obsA = self._observe_team('A')
        obsB = self._observe_team('B')

        info = {
            "step": int(self._step_count),
            "winner": winner,  # ★ 항상 'A' 또는 'B'
            "aliveA": (self.hpA > 0).astype(np.int32),
            "aliveB": (self.hpB > 0).astype(np.int32),
            "nexusA": self.nexusA.copy(),
            "nexusB": self.nexusB.copy(),
            "posA": self.posA.copy(),
            "posB": self.posB.copy(),
            "bullets": np.array([[b["x"], b["y"]] for b in self.bullets], dtype=np.float32),
            "shots": self.last_shots[:]  # 리스트(유니티 직렬화용)
        }

        return obsA, obsB, rA, rB, done, info
