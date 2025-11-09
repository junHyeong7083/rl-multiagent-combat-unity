import numpy as np
from commands import Command, onehot_cmd, N_COMMANDS

class CombatHierEnv:
    """
    계층형 멀티에이전트 전투 환경.
    - 유닛 행동: 0=stay,1=up,2=down,3=left,4=right,5=attack,6=regroup
    - 지휘 명령: ATTACK, DEFEND, HOLD, FLANK_L, FLANK_R, RETREAT
    - 관측:
      * 팀 전역(지휘관): [my_units(x,y,hp,alive)*N, en_units(...)*N, base_me, base_en]  (정규화)
      * 유닛 로컬(병사): [self(x,y,hp,alive), nearest Ally (dx,dy,dist), nearest Enemy (dx,dy,dist),
                         local_adv, my_cooldowns, cmd_onehot]
    - 보상:
      * (가한피해 - 받은피해) + 승패 보너스
      * 전술 셰이핑: 목표 진척(ATTACK/DEFEND/FLANK 등) + 위험 공격 패널티
    """
    def __init__(
        self,
        n_per_team: int = 5,
        arena_size: int = 16,
        max_steps: int = 300,
        move_cooldown: int = 0,
        attack_cooldown: int = 3,
        attack_range: float = 4.5,
        attack_damage: float = 0.3,
        win_bonus: float = 5.0,
        shaping_lambda_progress: float = 0.02,
        shaping_penalty_risky_attack: float = 0.05,
        local_adv_radius: float = 4.5,
        seed: int | None = None
    ):
        self.n = n_per_team
        self.S = arena_size
        self.max_steps = max_steps
        self.move_cd_max = move_cooldown
        self.atk_cd_max = attack_cooldown
        self.atk_range = attack_range
        self.atk_damage = attack_damage
        self.win_bonus = win_bonus
        self.sha_lam_prog = shaping_lambda_progress
        self.sha_pen_risky = shaping_penalty_risky_attack
        self.adv_R = local_adv_radius
        self.rng = np.random.RandomState(seed)

        self.t = 0
        self._alloc()

    # ---------- helpers ----------

    def _alloc(self):
        N = self.n
        self.posA = np.zeros((N, 2), dtype=np.float32)
        self.posB = np.zeros((N, 2), dtype=np.float32)
        self.hpA = np.ones((N,), dtype=np.float32)
        self.hpB = np.ones((N,), dtype=np.float32)
        self.aliveA = np.ones((N,), dtype=np.float32)
        self.aliveB = np.ones((N,), dtype=np.float32)
        self.move_cd_A = np.zeros((N,), dtype=np.int32)
        self.move_cd_B = np.zeros((N,), dtype=np.int32)
        self.atk_cd_A = np.zeros((N,), dtype=np.int32)
        self.atk_cd_B = np.zeros((N,), dtype=np.int32)
        self.baseA = 1.0
        self.baseB = 1.0

        # 지휘/집결 지점
        self.cmdA = Command.ATTACK
        self.cmdB = Command.DEFEND
        self.rallyA = np.array([2, self.S//2], dtype=np.float32)
        self.rallyB = np.array([self.S-3, self.S//2], dtype=np.float32)
        self.attack_targetA = np.array([self.S-2, self.S//2], dtype=np.float32)
        self.attack_targetB = np.array([1, self.S//2], dtype=np.float32)
        self._prev_progress_A = 0.0
        self._prev_progress_B = 0.0

    def _spawn(self):
        for i in range(self.n):
            self.posA[i] = np.array([self.rng.randint(1, 3), self.rng.randint(2, self.S-2)], dtype=np.float32)
            self.posB[i] = np.array([self.rng.randint(self.S-3, self.S-1), self.rng.randint(2, self.S-2)], dtype=np.float32)

    def _normalize_pos(self, p):
        return p / float(self.S - 1)

    def _team_obs_for(self, side: str):
        if side == "A":
            me_pos, me_hp, me_alive = self.posA, self.hpA, self.aliveA
            en_pos, en_hp, en_alive = self.posB, self.hpB, self.aliveB
            base_me, base_en = self.baseA, self.baseB
        else:
            me_pos, me_hp, me_alive = self.posB, self.hpB, self.aliveB
            en_pos, en_hp, en_alive = self.posA, self.hpA, self.aliveA
            base_me, base_en = self.baseB, self.baseA

        me = []
        for i in range(self.n):
            me.append(self._normalize_pos(me_pos[i]))
            me.append([me_hp[i], me_alive[i]])
        me = np.concatenate(me, axis=0)

        en = []
        for i in range(self.n):
            en.append(self._normalize_pos(en_pos[i]))
            en.append([en_hp[i], en_alive[i]])
        en = np.concatenate(en, axis=0)

        obs = np.concatenate([me, en, np.array([base_me, base_en], dtype=np.float32)], axis=0)
        return obs.astype(np.float32)

    def _nearest(self, src, tgt, tgt_alive):
        idx = np.where(tgt_alive > 0.5)[0]
        if idx.size == 0:
            return np.array([0,0], dtype=np.float32), 0.0, -1
        diffs = tgt[idx] - src
        d2 = np.sum(diffs*diffs, axis=1)
        jloc = np.argmin(d2)
        v = diffs[jloc]
        d = float(np.sqrt(d2[jloc]))
        return v, d, idx[jloc]

    def _local_advantage(self, p, allies_pos, allies_hp, allies_alive, enemies_pos, enemies_hp, enemies_alive, R):
        # 반경 R 내 체력합 차이
        idxA = np.where(allies_alive > 0.5)[0]
        idxE = np.where(enemies_alive > 0.5)[0]
        a = 0.0; e = 0.0
        for i in idxA:
            if np.linalg.norm(allies_pos[i]-p) <= R:
                a += float(allies_hp[i])
        for j in idxE:
            if np.linalg.norm(enemies_pos[j]-p) <= R:
                e += float(enemies_hp[j])
        return a - e  # >0 이면 우세

    def _in_bounds(self, p):
        p[0] = np.clip(p[0], 0, self.S-1)
        p[1] = np.clip(p[1], 0, self.S-1)
        return p

    def _move_dirs(self, a):
        dirs = {
            1: np.array([0, -1], dtype=np.float32),
            2: np.array([0,  1], dtype=np.float32),
            3: np.array([-1, 0], dtype=np.float32),
            4: np.array([1,  0], dtype=np.float32),
        }
        return dirs.get(int(a), None)

    # ---------- public API ----------

    def reset(self):
        self._alloc()
        self._spawn()
        self.t = 0
        self._prev_progress_A = self._progress("A")
        self._prev_progress_B = self._progress("B")
        return self._team_obs_for("A"), self._team_obs_for("B")

    def unit_obs(self, side: str, i: int, cmd_onehot_vec):
        if side == "A":
            my_pos, my_hp, my_alive = self.posA, self.hpA, self.aliveA
            al_pos, al_hp, al_alive = self.posA, self.hpA, self.aliveA
            en_pos, en_hp, en_alive = self.posB, self.hpB, self.aliveB
            move_cd, atk_cd = self.move_cd_A, self.atk_cd_A
        else:
            my_pos, my_hp, my_alive = self.posB, self.hpB, self.aliveB
            al_pos, al_hp, al_alive = self.posB, self.hpB, self.aliveB
            en_pos, en_hp, en_alive = self.posA, self.hpA, self.aliveA
            move_cd, atk_cd = self.move_cd_B, self.atk_cd_B

        p = my_pos[i]
        self_v_ally, self_d_ally, _ = self._nearest(p, al_pos, al_alive)
        self_v_enemy, self_d_enemy, _ = self._nearest(p, en_pos, en_alive)
        adv = self._local_advantage(p, al_pos, al_hp, al_alive, en_pos, en_hp, en_alive, self.adv_R)

        obs = np.concatenate([
            self._normalize_pos(p),                      # (2,)
            np.array([my_hp[i], my_alive[i]], np.float32),  # (2,)
            self_v_ally / max(1.0, self.adv_R),         # (2,)
            np.array([self_d_ally / max(1.0, self.S)], np.float32),  # (1,)
            self_v_enemy / max(1.0, self.adv_R),        # (2,)
            np.array([self_d_enemy / max(1.0, self.S)], np.float32), # (1,)
            np.array([adv], np.float32),                # (1,)
            np.array([move_cd[i] / max(1, self.move_cd_max+1),
                      atk_cd[i] / max(1, self.atk_cd_max+1)], np.float32),  # (2,)
            cmd_onehot_vec.astype(np.float32)           # (N_COMMANDS,)
        ], axis=0)
        return obs.astype(np.float32)

    @property
    def unit_obs_dim(self):
        # 2(self pos) +2(hp,alive) +2+1(ally vec,dist) +2+1(enemy vec,dist) +1(adv) +2(cds) + N_COMMANDS
        return 2+2 +3 +3 +1 +2 + N_COMMANDS

    @property
    def team_obs_dim(self):
        # team-centric like before:  (x,y,hp,alive)*N *2 teams + 2 base scalars
        return self.n*4*2 + 2

    @property
    def n_actions_unit(self):
        return 7

    def _progress(self, side: str):
        # ATTACK 관점의 전진 정도: 내 유닛들이 목표 방향으로 얼마나 가까운지(평균)
        if side == "A":
            tgt = self.attack_targetA
            pos = self.posA
            alive = self.aliveA
        else:
            tgt = self.attack_targetB
            pos = self.posB
            alive = self.aliveB
        idx = np.where(alive > 0.5)[0]
        if idx.size == 0: return 0.0
        d = 0.0
        for i in idx:
            d += float(np.linalg.norm(pos[i]-tgt))
        d /= idx.size
        # 거리가 작을수록 좋은데, 보상을 +로 만들기 위해 음수부호
        return -d

    def step(self, actA, actB):
        self.t += 1
        # movement
        self._step_move_block(self.posA, self.aliveA, self.move_cd_A, actA, self.rallyA)
        self._step_move_block(self.posB, self.aliveB, self.move_cd_B, actB, self.rallyB)

        # attack
        dmgA_to_B, dmgB_to_A = self._step_attack_blocks(actA, actB)

        # base reward
        rA = float(dmgA_to_B - dmgB_to_A)
        rB = float(dmgB_to_A - dmgA_to_B)

        # shaping: progress
        progA = self._progress("A")
        progB = self._progress("B")
        rA += self.sha_lam_prog * (progA - self._prev_progress_A)
        rB += self.sha_lam_prog * (progB - self._prev_progress_B)
        self._prev_progress_A, self._prev_progress_B = progA, progB

        # risky attack penalty
        rA -= self._risky_attack_penalty("A", actA)
        rB -= self._risky_attack_penalty("B", actB)

        done, winner = self._terminal_and_winner()
        if done and winner != "draw":
            if winner == "A":
                rA += self.win_bonus; rB -= self.win_bonus
            elif winner == "B":
                rB += self.win_bonus; rA -= self.win_bonus

        info = {
            "aliveA": self.aliveA.copy(),
            "aliveB": self.aliveB.copy(),
            "winner": winner if done else None
        }
        return self._team_obs_for("A"), self._team_obs_for("B"), rA, rB, done, info

    def _step_move_block(self, pos, alive, move_cd, actions, rally_point):
        for i in range(self.n):
            if alive[i] <= 0.5: continue
            if move_cd[i] > 0:
                move_cd[i] -= 1
                continue
            a = int(actions[i])
            if a == 6:  # regroup
                v = rally_point - pos[i]
                if np.linalg.norm(v) > 1e-6:
                    v = v / np.linalg.norm(v)
                pos[i] = self._in_bounds(pos[i] + v)
                if self.move_cd_max > 0: move_cd[i] = self.move_cd_max
                continue
            d = self._move_dirs(a)
            if d is not None:
                pos[i] = self._in_bounds(pos[i] + d)
                if self.move_cd_max > 0: move_cd[i] = self.move_cd_max

    def _step_attack_blocks(self, actA, actB):
        dmgA_to_B = self._attack_side(self.posA, self.aliveA, self.hpA, self.atk_cd_A,
                                      self.posB, self.aliveB, self.hpB, actA)
        dmgB_to_A = self._attack_side(self.posB, self.aliveB, self.hpB, self.atk_cd_B,
                                      self.posA, self.aliveA, self.hpA, actB)
        return dmgA_to_B, dmgB_to_A

    def _attack_side(self, my_pos, my_alive, my_hp, my_atk_cd, en_pos, en_alive, en_hp, actions):
        dmg = 0.0
        for i in range(self.n):
            if my_alive[i] <= 0.5: continue
            if my_atk_cd[i] > 0:
                my_atk_cd[i] -= 1
                continue
            if int(actions[i]) != 5:  # attack
                continue
            idx = np.where(en_alive > 0.5)[0]
            if idx.size == 0: continue
            diffs = en_pos[idx] - my_pos[i]
            d2 = np.sum(diffs*diffs, axis=1)
            jloc = np.argmin(d2)
            dist = float(np.sqrt(d2[jloc]))
            if dist <= self.atk_range:
                j = idx[jloc]
                en_hp[j] -= self.atk_damage
                dmg += self.atk_damage
                my_atk_cd[i] = self.atk_cd_max
        # clamp & alive update
        en_hp[:] = np.clip(en_hp, 0.0, 1.0)
        for i in range(self.n):
            if en_hp[i] <= 0.0: en_alive[i] = 0.0
            if my_hp[i] <= 0.0: my_alive[i] = 0.0
        return dmg

    def _risky_attack_penalty(self, side: str, actions):
        if side == "A":
            my_pos, my_hp, my_alive = self.posA, self.hpA, self.aliveA
            al_pos, al_hp, al_alive = self.posA, self.hpA, self.aliveA
            en_pos, en_hp, en_alive = self.posB, self.hpB, self.aliveB
        else:
            my_pos, my_hp, my_alive = self.posB, self.hpB, self.aliveB
            al_pos, al_hp, al_alive = self.posB, self.hpB, self.aliveB
            en_pos, en_hp, en_alive = self.posA, self.hpA, self.aliveA
        pen = 0.0
        for i in range(self.n):
            if my_alive[i] <= 0.5: continue
            if int(actions[i]) == 5:  # attack
                adv = self._local_advantage(my_pos[i], al_pos, al_hp, al_alive, en_pos, en_hp, en_alive, self.adv_R)
                if adv < 0.0:
                    pen += self.sha_pen_risky
        return pen

    def _terminal_and_winner(self):
        a_alive = int(np.sum(self.aliveA > 0.5))
        b_alive = int(np.sum(self.aliveB > 0.5))
        if a_alive == 0 and b_alive == 0: return True, "draw"
        if a_alive == 0: return True, "B"
        if b_alive == 0: return True, "A"
        if self.t >= self.max_steps:
            sA = float(np.sum(self.hpA)); sB = float(np.sum(self.hpB))
            if abs(sA - sB) < 1e-6: return True, "draw"
            return True, "A" if sA > sB else "B"
        return False, None
