import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from multiagent_env_hier import CombatSelfPlayHierEnv
from ac_models import UnitActorCritic, CommanderActorCritic
from commands import Command, N_COMMANDS, onehot_cmd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- 하이퍼파라미터 -----------------
SEED          = 123
N_PER_TEAM    = 5
HORIZON       = 128
GAMMA         = 0.98
LR_UNIT       = 3e-4
LR_CMD        = 3e-4
VF_COEF       = 0.5
ENT_COEF      = 0.01
GRAD_CLIP     = 0.5

EPOCHS        = 2000
FREEZE_K      = 20
SAVE_INT      = 50
EVAL_INT      = 100
CMD_INTERVAL  = 8

CKPT_DIR      = "./ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)

def discounted_returns(x, gamma=GAMMA):
    out = np.zeros_like(x, dtype=np.float32)
    run = 0.0
    for t in reversed(range(len(x))):
        run = x[t] + gamma * run
        out[t] = run
    return out

@torch.no_grad()
def policy_action(logits):
    p = torch.softmax(logits, dim=-1)
    a = torch.argmax(p, dim=-1)
    return a

def clone_state(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def load_state(model: nn.Module, sd):
    model.load_state_dict(sd)

def rollout(env: CombatSelfPlayHierEnv,
            unitA: UnitActorCritic, unitB: UnitActorCritic,
            cmdA: CommanderActorCritic, cmdB: CommanderActorCritic,
            train_side: str, cmd_interval: int = CMD_INTERVAL):

    obsA, obsB = env.reset()
    T = HORIZON

    # 커맨더 상태
    cur_cmdA = Command.ATTACK
    cur_cmdB = Command.DEFEND
    cmdA_hist, cmdB_hist = [], []

    obsA_buf, obsB_buf = [], []
    actA_buf, actB_buf = [], []
    rA_buf, rB_buf     = [], []
    done_buf           = []

    for t in range(T):
        # 커맨더 갱신
        if t % cmd_interval == 0:
            # 팀 평균 관측을 대표로 커맨더에 입력 (여기선 유닛 관측 첫개를 사용)
            oA_cmd = torch.from_numpy(obsA[0:1]).float().to(device)
            oB_cmd = torch.from_numpy(obsB[0:1]).float().to(device)
            with torch.no_grad():
                logitA, _ = cmdA(oA_cmd)
                logitB, _ = cmdB(oB_cmd)
                cur_cmdA = int(policy_action(logitA)[0].item())
                cur_cmdB = int(policy_action(logitB)[0].item())

        ohA = onehot_cmd(cur_cmdA)  # (6,)
        ohB = onehot_cmd(cur_cmdB)

        # 유닛 입력: 각 유닛 관측에 커맨더 1-hot을 붙인다
        obsA_17 = np.concatenate([obsA, np.tile(ohA[None, :], (env.n,1))], axis=1)
        obsB_17 = np.concatenate([obsB, np.tile(ohB[None, :], (env.n,1))], axis=1)

        obsA_buf.append(obsA_17.copy())
        obsB_buf.append(obsB_17.copy())
        cmdA_hist.append(cur_cmdA)
        cmdB_hist.append(cur_cmdB)

        # 유닛 행동
        oA_t = torch.from_numpy(obsA_17).float().to(device)
        oB_t = torch.from_numpy(obsB_17).float().to(device)
        with torch.no_grad():
            logitsA, _ = unitA(oA_t)
            logitsB, _ = unitB(oB_t)
            aA = policy_action(logitsA).cpu().numpy()
            aB = policy_action(logitsB).cpu().numpy()

        actA_buf.append(aA.copy())
        actB_buf.append(aB.copy())

        obsA, obsB, rA, rB, done, info = env.step(aA, aB)

        # 팀 보상은 합계로 스케일업
        rA_buf.append(float(np.sum(rA)))
        rB_buf.append(float(np.sum(rB)))
        done_buf.append(done)
        if done:
            break

    data = {
        "obsA": np.asarray(obsA_buf, dtype=np.float32),         # (T, N, D+6)
        "obsB": np.asarray(obsB_buf, dtype=np.float32),
        "actA": np.asarray(actA_buf, dtype=np.int64),           # (T, N)
        "actB": np.asarray(actB_buf, dtype=np.int64),
        "rA":   np.asarray(rA_buf, dtype=np.float32),           # (T,)
        "rB":   np.asarray(rB_buf, dtype=np.float32),
        "cmdA": np.asarray(cmdA_hist, dtype=np.int64),          # (T,)
        "cmdB": np.asarray(cmdB_hist, dtype=np.int64),
        "done": np.asarray(done_buf, dtype=np.bool_),
        "final_info": info,
        "steps": len(rA_buf),
    }
    return data

def train_one_side(unit: UnitActorCritic, cmd: CommanderActorCritic,
                   data, gamma=GAMMA):
    # 유닛 손실
    obs = torch.tensor(data["obs"], dtype=torch.float32, device=device)  # (T,N,D+6)
    act = torch.tensor(data["act"], dtype=torch.int64,   device=device)  # (T,N)
    rets= torch.tensor(discounted_returns(data["ret"], gamma), dtype=torch.float32, device=device)  # (T,)

    T, N, D = obs.shape
    obs = obs.view(T*N, D)
    act = act.view(T*N)

    logits, v_pred = unit(obs)
    logp = torch.log_softmax(logits, dim=-1)
    act_logp = logp[torch.arange(T*N), act]

    v_tgt = rets[:, None].repeat(1, N).reshape(T*N)
    adv   = (v_tgt - v_pred.detach())

    pi_loss = -(act_logp * adv).mean()
    v_loss  = 0.5 * (v_pred - v_tgt).pow(2).mean()
    ent     = -(logp * torch.exp(logp)).sum(dim=-1).mean()
    unit_loss = pi_loss + VF_COEF * v_loss - ENT_COEF * ent

    # 커맨더 손실(에피소드 타임스텝당 1개)
    # 커맨더 입력은 obs의 첫 유닛(대표)에서 커맨더 one-hot 제외한 앞단 D-6을 사용
    D_total = D
    D_base  = D_total - N_COMMANDS
    obs_cmd = obs.view(T, N, D_total)[:,0,:D_base]                   # (T, D_base)
    cmd_idx = torch.tensor(data["cmd"], dtype=torch.int64, device=device)  # (T,)

    logits_c, v_c = cmd(obs_cmd)
    logp_c = torch.log_softmax(logits_c, dim=-1)
    act_logp_c = logp_c[torch.arange(T), cmd_idx]

    v_tgt_c = rets  # 같은 리턴 신호 공유
    adv_c   = (v_tgt_c - v_c.detach())

    pi_loss_c = -(act_logp_c * adv_c).mean()
    v_loss_c  = 0.5 * (v_c - v_tgt_c).pow(2).mean()
    ent_c     = -(logp_c * torch.exp(logp_c)).sum(dim=-1).mean()
    cmd_loss  = pi_loss_c + VF_COEF * v_loss_c - ENT_COEF * ent_c

    return unit_loss, cmd_loss

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    def mk_env(seed=None):
        return CombatSelfPlayHierEnv(n_per_team=N_PER_TEAM, seed=(seed if seed is not None else np.random.randint(1<<31)))
    env = mk_env(SEED)

    obs_base = env.obs_dim
    obs_total= obs_base + N_COMMANDS

    unitA = UnitActorCritic(obs_total, env.n_actions, hidden=256).to(device)
    unitB = UnitActorCritic(obs_total, env.n_actions, hidden=256).to(device)
    cmdA  = CommanderActorCritic(obs_base, N_COMMANDS, hidden=256).to(device)
    cmdB  = CommanderActorCritic(obs_base, N_COMMANDS, hidden=256).to(device)

    optA_u = optim.Adam(unitA.parameters(), lr=LR_UNIT)
    optB_u = optim.Adam(unitB.parameters(), lr=LR_UNIT)
    optA_c = optim.Adam(cmdA.parameters(),  lr=LR_CMD)
    optB_c = optim.Adam(cmdB.parameters(),  lr=LR_CMD)

    for ep in range(1, EPOCHS+1):
        train_A = ((ep // FREEZE_K) % 2 == 0)

        if train_A:
            unitB.eval(); cmdB.eval()
            unitA.train(); cmdA.train()
        else:
            unitA.eval(); cmdA.eval()
            unitB.train(); cmdB.train()

        data = rollout(env, unitA, unitB, cmdA, cmdB, 'A' if train_A else 'B', CMD_INTERVAL)

        if train_A:
            pack = {"obs": data["obsA"], "act": data["actA"], "ret": data["rA"], "cmd": data["cmdA"]}
            u_loss, c_loss = train_one_side(unitA, cmdA, pack)
            optA_u.zero_grad(); u_loss.backward(); nn.utils.clip_grad_norm_(unitA.parameters(), GRAD_CLIP); optA_u.step()
            optA_c.zero_grad(); c_loss.backward(); nn.utils.clip_grad_norm_(cmdA.parameters(),  GRAD_CLIP); optA_c.step()
        else:
            pack = {"obs": data["obsB"], "act": data["actB"], "ret": data["rB"], "cmd": data["cmdB"]}
            u_loss, c_loss = train_one_side(unitB, cmdB, pack)
            optB_u.zero_grad(); u_loss.backward(); nn.utils.clip_grad_norm_(unitB.parameters(), GRAD_CLIP); optB_u.step()
            optB_c.zero_grad(); c_loss.backward(); nn.utils.clip_grad_norm_(cmdB.parameters(),  GRAD_CLIP); optB_c.step()

        if ep % 10 == 0:
            print(f"ep {ep:04d} | train={'A' if train_A else 'B'} | unit_loss={u_loss.item():.3f} | cmd_loss={c_loss.item():.3f} | retA={np.mean(data['rA']):.2f} retB={np.mean(data['rB']):.2f}")

        if ep % SAVE_INT == 0:
            torch.save(unitA.state_dict(), os.path.join(CKPT_DIR, f"a_unit_ep{ep:04d}.pt"))
            torch.save(unitB.state_dict(), os.path.join(CKPT_DIR, f"b_unit_ep{ep:04d}.pt"))
            torch.save(cmdA.state_dict(),  os.path.join(CKPT_DIR, f"a_cmd_ep{ep:04d}.pt"))
            torch.save(cmdB.state_dict(),  os.path.join(CKPT_DIR, f"b_cmd_ep{ep:04d}.pt"))

if __name__ == "__main__":
    main()
