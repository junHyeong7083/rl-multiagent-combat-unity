import numpy as np

class CombatSelfPlay5v5Env:
    ACT_STAY = 0
    ACT_MOVE_DIRS = {
        1: (0, -1), 2: (0,  1), 3: (-1, 0), 4: (1,  0),
        5: (-1,-1), 6: (1, -1), 7: (-1, 1), 8: (1,  1),
    }
    ACT_ATTACK = 9

    def __init__(self, width=32, height=32, n_per_team=5, max_steps=180,
                 hp=3, reload_steps=5, seed=None):
        self.width = int(width)
        self.height = int(height)
        self.n = int(n_per_team)
        self.max_steps = int(max_steps)
        self.hp_max = int(hp)
        self.reload_steps = int(reload_steps)
        self.rng = np.random.RandomState(seed)
        self.A = None
        self.B = None
        self.t = 0
        self.base_A = (1, self.height-2)   # ~7시 (SW)
        self.base_B = (self.width-2, 1)    # ~1시 (NE)
        self._init_state()

    def _spawn_cluster(self, base, count, radius=2):
        bx, by = base
        pts = []
        tries = 0
        while len(pts) < count and tries < count*20:
            tries += 1
            dx = self.rng.randint(-radius, radius+1)
            dy = self.rng.randint(-radius, radius+1)
            x = np.clip(bx+dx, 1, self.width-2)
            y = np.clip(by+dy, 1, self.height-2)
            if (x,y) not in pts:
                pts.append((x,y))
        while len(pts) < count:
            x = self.rng.randint(1, self.width-1)
            y = self.rng.randint(1, self.height-1)
            if (x,y) not in pts:
                pts.append((x,y))
        return pts

    def _init_state(self):
        self.t = 0
        # agents columns: x,y,hp, fx,fy, cooldown
        self.A = np.zeros((self.n, 6), dtype=np.int32)
        self.B = np.zeros((self.n, 6), dtype=np.int32)
        a_pts = self._spawn_cluster(self.base_A, self.n)
        b_pts = self._spawn_cluster(self.base_B, self.n)
        for i,(x,y) in enumerate(a_pts):
            self.A[i,:] = (x,y,self.hp_max, 1,0, 0)
        for i,(x,y) in enumerate(b_pts):
            self.B[i,:] = (x,y,self.hp_max, -1,0, 0)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._init_state()
        return (self._obs_team(self.A, self.B),
                self._obs_team(self.B, self.A))

    def _alive_mask(self, arr):
        return arr[:,2] > 0

    def _in_bounds(self, x,y):
        return 0 <= x < self.width and 0 <= y < self.height

    def _move_one(self, agent, dir_id, occ):  # occ: occupancy set to avoid overlap
        if agent[2] <= 0: return
        if dir_id == self.ACT_STAY: return
        dx,dy = self.ACT_MOVE_DIRS[dir_id]
        nx,ny = agent[0]+dx, agent[1]+dy
        if not self._in_bounds(nx,ny): return
        if (nx,ny) in occ: return
        agent[0]=nx; agent[1]=ny; agent[3]=np.sign(dx) if dx!=0 else 0; agent[4]=np.sign(dy) if dy!=0 else 0

    def _line_hit(self, shooter, enemies):
        sx, sy, _, fx, fy, _ = shooter
        if fx==0 and fy==0: return -1, None
        fx = 0 if fx==0 else (1 if fx>0 else -1)
        fy = 0 if fy==0 else (1 if fy>0 else -1)
        best = None
        best_d = 10**9
        for idx,(ex,ey,ehp, *_rest) in enumerate(enemies):
            if ehp<=0: continue
            dx = ex - sx; dy = ey - sy
            # inline check
            if fx==0 and dy*np.sign(fy)>=0 and dx==0 and (fy!=0):
                d = abs(dy)
            elif fy==0 and dx*np.sign(fx)>=0 and dy==0 and (fx!=0):
                d = abs(dx)
            elif abs(dx)==abs(dy) and (np.sign(dx)==fx) and (np.sign(dy)==fy):
                d = abs(dx)
            else:
                continue
            if d < best_d:
                best_d = d
                best = idx
        if best is None:
            # for viz endpoint ray
            end = (np.clip(sx+fx*max(self.width,self.height),0,self.width-1),
                   np.clip(sy+fy*max(self.width,self.height),0,self.height-1))
            return -1, end
        return best, (enemies[best,0], enemies[best,1])

    def step(self, actA, actB):
        self.t += 1
        shots = []
        # movement occupancy
        occA = {(x,y) for (x,y,hp,fx,fy,cd) in self.A if hp>0}
        occB = {(x,y) for (x,y,hp,fx,fy,cd) in self.B if hp>0}
        # move
        for i,a in enumerate(self.A):
            if a[2]<=0: continue
            if actA[i] in self.ACT_MOVE_DIRS:
                occA.remove((a[0],a[1]))
                self._move_one(a, actA[i], occA)
                occA.add((a[0],a[1]))
        for i,b in enumerate(self.B):
            if b[2]<=0: continue
            if actB[i] in self.ACT_MOVE_DIRS:
                occB.remove((b[0],b[1]))
                self._move_one(b, actB[i], occB)
                occB.add((b[0],b[1]))
        # shooting
        def do_attacks(me, other, tag):
            r_me = 0
            for i,ag in enumerate(me):
                if ag[2]<=0: continue
                if ag[5]>0:
                    ag[5]-=1
                    continue
                if (tag=='A' and actA[i]==self.ACT_ATTACK) or (tag=='B' and actB[i]==self.ACT_ATTACK):
                    hit_idx, end = self._line_hit(ag, other)
                    sx,sy = ag[0],ag[1]
                    if hit_idx>=0:
                        ox,oy = other[hit_idx,0], other[hit_idx,1]
                        other[hit_idx,2] -= 1
                        shots.append({"from":[int(sx),int(sy)],"to":[int(ox),int(oy)],"hit":True,"team":tag})
                        ag[5]=self.reload_steps
                        r_me += 0.2
                    else:
                        tx,ty = end if end is not None else (sx,sy)
                        shots.append({"from":[int(sx),int(sy)],"to":[int(tx),int(ty)],"hit":False,"team":tag})
                        ag[5]=self.reload_steps
            return r_me
        rA = do_attacks(self.A, self.B, 'A')
        rB = do_attacks(self.B, self.A, 'B')

        # base capture instant win
        capA = any((x,y)==self.base_B for (x,y,hp,fx,fy,cd) in self.A if hp>0)
        capB = any((x,y)==self.base_A for (x,y,hp,fx,fy,cd) in self.B if hp>0)

        done=False
        if capA and not capB:
            done=True; rA += 10.0; rB -= 10.0
            outcome = 'A_capture'
        elif capB and not capA:
            done=True; rA -= 10.0; rB += 10.0
            outcome = 'B_capture'
        else:
            aliveA = self._alive_mask(self.A).sum()
            aliveB = self._alive_mask(self.B).sum()
            if aliveA==0 and aliveB>0:
                done=True; rA -= 5.0; rB += 5.0; outcome='B_wipe'
            elif aliveB==0 and aliveA>0:
                done=True; rA += 5.0; rB -= 5.0; outcome='A_wipe'
            elif self.t>=self.max_steps:
                done=True; outcome='timeout'
            else:
                outcome=None

        info = {"shots": shots, "base_A": self.base_A, "base_B": self.base_B, "outcome": outcome}
        return self._obs_team(self.A, self.B), self._obs_team(self.B, self.A), float(rA), float(rB), done, info

    def _obs_team(self, me, other):
        return np.concatenate([me.flatten(), other.flatten()]).astype(np.int32)

    def sample_actions(self):
        return self.rng.randint(0, 10, size=self.n, dtype=np.int32)
