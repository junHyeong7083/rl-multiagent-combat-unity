import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from commands import Command, onehot_cmd, N_COMMANDS
from ac_models import CommanderPolicy, UnitPolicy, TeamValueNet
from multiagent_env_hier import CombatHierEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def masked_mean(x, mask, eps=1e-8):
    num = (x * mask).sum()
    den = mask.sum().clamp_min(eps)
    return num / den

def rollout(env, cmdA, cmdB, K_cmd, unit_pi, vnet, cmd_pi, horizon=128, gamma=0.99):
    obsA_team, obsB_team = env.reset()

    # buffers
    # commander (decision at t%K==0)
    cmd_logp_A, cmd_val_A, cmd_time_A = [], [], []
    cmd_logp_B, cmd_val_B, cmd_time_B = [], [], []

    # unit-level
    logpA_units, logpB_units = [], []
    aliveA_buf, aliveB_buf = [], []
    valA_team, valB_team = [], []
    rewA, rewB, done_buf = [], [], []
    obsA_team_buf, obsB_team_buf = [], []

    for t in range(horizon):
        oA_team_t = torch.from_numpy(obsA_team).float().to(device)
        oB_team_t = torch.from_numpy(obsB_team).float().to(device)

        # commander decides every K steps
        if t % K_cmd == 0:
            logits_cmd_A, v_cmd_A = cmdA(oA_team_t); pA = torch.softmax(logits_cmd_A, dim=-1)
            distA = torch.distributions.Categorical(probs=pA)
            cA = distA.sample()
            logp_cA = distA.log_prob(cA)

            logits_cmd_B, v_cmd_B = cmdB(oB_team_t); pB = torch.softmax(logits_cmd_B, dim=-1)
            distB = torch.distributions.Categorical(probs=pB)
            cB = distB.sample()
            logp_cB = distB.log_prob(cB)

            curr_cmdA = int(cA.item())
            curr_cmdB = int(cB.item())

            cmd_logp_A.append(logp_cA); cmd_val_A.append(v_cmd_A); cmd_time_A.append(t)
            cmd_logp_B.append(logp_cB); cmd_val_B.append(v_cmd_B); cmd_time_B.append(t)
        # keep last command otherwise
        cmdA_id = curr_cmdA
        cmdB_id = curr_cmdB

        # team value (baseline for unit policy)
        vA = vnet(oA_team_t); vB = vnet(oB_team_t)
        valA_team.append(vA); valB_team.append(vB)
        obsA_team_buf.append(oA_team_t); obsB_team_buf.append(oB_team_t)

        # per-unit actions
        aA_np = np.zeros((env.n,), dtype=np.int64)
        aB_np = np.zeros((env.n,), dtype=np.int64)

        cmdA_onehot = onehot_cmd(cmdA_id)
        cmdB_onehot = onehot_cmd(cmdB_id)

        logpAu = []
        logpBu = []
        aliveA_now = []
        aliveB_now = []

        for i in range(env.n):
            # side A
            uoA = env.unit_obs("A", i, cmdA_onehot)
            uoA_t = torch.from_numpy(uoA).float().to(device)
            logitsA = unit_pi(uoA_t); pA_u = torch.softmax(logitsA, dim=-1)
            dA_u = torch.distributions.Categorical(probs=pA_u)
            aA = dA_u.sample(); logpAu.append(dA_u.log_prob(aA))
            aA_np[i] = int(aA.item())
            aliveA_now.append(float(uoA[1+1]))  # alive index: [pos(2), hp,alive]-> alive at idx=3 -> here 1+1? quick explicit:
            # 정확히 하자: unit_obs에서 [pos2, hp, alive] 순서이므로 alive는 index 3.
            aliveA_now[-1] = float(uoA[3])

            # side B
            uoB = env.unit_obs("B", i, cmdB_onehot)
            uoB_t = torch.from_numpy(uoB).float().to(device)
            logitsB = unit_pi(uoB_t); pB_u = torch.softmax(logitsB, dim=-1)
            dB_u = torch.distributions.Categorical(probs=pB_u)
            aB = dB_u.sample(); logpBu.append(dB_u.log_prob(aB))
            aB_np[i] = int(aB.item())
            aliveB_now.append(float(uoB[3]))

        obsA_team_next, obsB_team_next, rA, rB, done, info = env.step(aA_np, aB_np)

        logpA_units.append(torch.stack(logpAu))  # [n]
        logpB_units.append(torch.stack(logpBu))  # [n]
        aliveA_buf.append(torch.tensor(aliveA_now, dtype=torch.float32, device=device))
        aliveB_buf.append(torch.tensor(aliveB_now, dtype=torch.float32, device=device))
        rewA.append(torch.tensor(rA, dtype=torch.float32, device=device))
        rewB.append(torch.tensor(rB, dtype=torch.float32, device=device))
        done_buf.append(torch.tensor(float(done), dtype=torch.float32, device=device))

        obsA_team = obsA_team_next
        obsB_team = obsB_team_next
        if done:
            break

    # returns
    T = len(rewA)
    retA = []; retB = []
    RA = torch.tensor(0.0, device=device); RB = torch.tensor(0.0, device=device)
    for t in reversed(range(T)):
        RA = rewA[t] + 0.99 * RA * (1.0 - done_buf[t])
        RB = rewB[t] + 0.99 * RB * (1.0 - done_buf[t])
        retA.append(RA); retB.append(RB)
    retA.reverse(); retB.reverse()

    data = {
        "T": T,
        "obsA_team": torch.stack(obsA_team_buf),
        "obsB_team": torch.stack(obsB_team_buf),
        "valA": torch.stack(valA_team),
        "valB": torch.stack(valB_team),
        "retA": torch.stack(retA),
        "retB": torch.stack(retB),
        "logpA_units": torch.stack(logpA_units),   # [T, n]
        "logpB_units": torch.stack(logpB_units),   # [T, n]
        "aliveA": torch.stack(aliveA_buf),         # [T, n]
        "aliveB": torch.stack(aliveB_buf),         # [T, n]
        "cmd_logp_A": cmd_logp_A,
        "cmd_val_A": cmd_val_A,
        "cmd_time_A": cmd_time_A,
        "cmd_logp_B": cmd_logp_B,
        "cmd_val_B": cmd_val_B,
        "cmd_time_B": cmd_time_B,
        "done": done_buf[-1].item() if T>0 else 1.0
    }
    return data

def train():
    env = CombatHierEnv(
        n_per_team=5,
        arena_size=16,
        max_steps=300,
        move_cooldown=0,
        attack_cooldown=3,
        attack_range=4.5,
        attack_damage=0.3,
        win_bonus=6.0,
        seed=42
    )

    K_cmd = 8  # 지휘관 명령 주기
    unit_pi = UnitPolicy(env.unit_obs_dim, env.n_actions_unit, hidden=256).to(device)
    vnet = TeamValueNet(env.team_obs_dim, hidden=256).to(device)
    cmdA = CommanderPolicy(env.team_obs_dim, N_COMMANDS, hidden=256).to(device)
    cmdB = CommanderPolicy(env.team_obs_dim, N_COMMANDS, hidden=256).to(device)

    opt = optim.Adam(list(unit_pi.parameters()) + list(vnet.parameters())
                     + list(cmdA.parameters()) + list(cmdB.parameters()),
                     lr=3e-4)

    max_epochs = 2000
    horizon = 128
    gamma = 0.99
    ent_coef_units = 0.01
    ent_coef_cmd = 0.005
    vf_coef = 0.5
    max_grad_norm = 0.5

    os.makedirs("ckpt", exist_ok=True)

    for ep in range(1, max_epochs+1):
        unit_pi.train(); vnet.train(); cmdA.train(); cmdB.train()

        data = rollout(env, cmdA, cmdB, K_cmd, unit_pi, vnet, cmdA, horizon=horizon, gamma=gamma)
        T = data["T"]
        if T == 0: continue

        # team advantage (공유) -> unit policy에 사용
        valA = data["valA"]; retA = data["retA"]; advA = retA - valA
        valB = data["valB"]; retB = data["retB"]; advB = retB - valB

        advA_rep = advA.unsqueeze(-1).repeat(1, env.n)
        advB_rep = advB.unsqueeze(-1).repeat(1, env.n)

        # unit policy loss (alive mask)
        logpA_u = data["logpA_units"]; logpB_u = data["logpB_units"]
        aliveA = data["aliveA"]; aliveB = data["aliveB"]
        pi_loss_A = -masked_mean(logpA_u * advA_rep.detach(), aliveA)
        pi_loss_B = -masked_mean(logpB_u * advB_rep.detach(), aliveB)
        pi_loss = 0.5 * (pi_loss_A + pi_loss_B)

        # unit entropy (approx: 평균 엔트로피 추정 위해 몇 스텝 샘플)
        with torch.no_grad():
            pass  # 간단화를 위해 생략 가능
        ent_units = 0.0  # 필요시 rollout 중 probs 저장해 사용

        # team value loss
        v_loss = 0.5 * (F.mse_loss(valA, retA) + F.mse_loss(valB, retB))

        # commander loss (K-step 시점들만)
        cmd_loss_pi = torch.tensor(0.0, device=device)
        cmd_loss_v = torch.tensor(0.0, device=device)
        if len(data["cmd_time_A"])>0:
            adv_cmdA = []
            for t, v in zip(data["cmd_time_A"], data["cmd_val_A"]):
                adv_cmdA.append((retA[t] - v).detach())
            adv_cmdA = torch.stack(adv_cmdA)
            logp_cmdA = torch.stack(data["cmd_logp_A"])
            v_cmdA = torch.stack(data["cmd_val_A"])
            cmd_loss_pi = cmd_loss_pi + (- (logp_cmdA * adv_cmdA).mean())
            cmd_loss_v = cmd_loss_v + F.mse_loss(v_cmdA, torch.stack([retA[t] for t in data["cmd_time_A"]]))

        if len(data["cmd_time_B"])>0:
            adv_cmdB = []
            for t, v in zip(data["cmd_time_B"], data["cmd_val_B"]):
                adv_cmdB.append((retB[t] - v).detach())
            adv_cmdB = torch.stack(adv_cmdB)
            logp_cmdB = torch.stack(data["cmd_logp_B"])
            v_cmdB = torch.stack(data["cmd_val_B"])
            cmd_loss_pi = cmd_loss_pi + (- (logp_cmdB * adv_cmdB).mean())
            cmd_loss_v = cmd_loss_v + F.mse_loss(v_cmdB, torch.stack([retB[t] for t in data["cmd_time_B"]]))

        loss = pi_loss + vf_coef*v_loss + cmd_loss_pi + 0.5*cmd_loss_v - ent_coef_units*0.0 - ent_coef_cmd*0.0

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(unit_pi.parameters())+list(vnet.parameters())
                                       +list(cmdA.parameters())+list(cmdB.parameters()), max_grad_norm)
        opt.step()

        if ep % 10 == 0:
            print(f"ep {ep:04d} | loss={loss.item():.3f} | pi_units={pi_loss.item():.3f} v={v_loss.item():.3f} "
                  f"| cmd_pi={cmd_loss_pi.item():.3f} cmd_v={cmd_loss_v.item():.3f} "
                  f"| retA={float(retA[0]):.2f} retB={float(retB[0]):.2f}")

        if ep % 200 == 0:
            torch.save({
                "unit_pi": unit_pi.state_dict(),
                "vnet": vnet.state_dict(),
                "cmdA": cmdA.state_dict(),
                "cmdB": cmdB.state_dict()
            }, f"ckpt/hier_{ep}.pt")

if __name__ == "__main__":
    train()
