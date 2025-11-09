import argparse, json, socket, time
import torch
import numpy as np

from commands import onehot_cmd, N_COMMANDS, Command
from ac_models import CommanderPolicy, UnitPolicy, TeamValueNet
from multiagent_env_hier import CombatHierEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pack_frame(env, step, info, rA, rB, cmdA_id, cmdB_id):
    def team_units(pos, hp, alive):
        return [
            {"x": float(pos[i][0]), "y": float(pos[i][1]),
             "hp": float(hp[i]), "alive": bool(alive[i] > 0.5)}
            for i in range(env.n)
        ]
    return {
        "step": step,
        "arena_size": env.S,
        "teamA": team_units(env.posA, env.hpA, env.aliveA),
        "teamB": team_units(env.posB, env.hpB, env.aliveB),
        "reward": {"A": float(rA), "B": float(rB)},
        "cmdA": int(cmdA_id),
        "cmdB": int(cmdB_id),
        "winner": info.get("winner"),
        "done": info.get("winner") is not None
    }

def eval_stream(model_path: str | None, host: str, port: int, fps: int, episodes: int, K_cmd: int = 8):
    env = CombatHierEnv(n_per_team=5, seed=123)
    unit_pi = UnitPolicy(env.unit_obs_dim, env.n_actions_unit, hidden=256).to(device)
    vnet = TeamValueNet(env.team_obs_dim, hidden=256).to(device)
    cmdA = CommanderPolicy(env.team_obs_dim, N_COMMANDS, hidden=256).to(device)
    cmdB = CommanderPolicy(env.team_obs_dim, N_COMMANDS, hidden=256).to(device)

    if model_path:
        sd = torch.load(model_path, map_location=device)
        unit_pi.load_state_dict(sd["unit_pi"])
        vnet.load_state_dict(sd["vnet"])
        cmdA.load_state_dict(sd["cmdA"])
        cmdB.load_state_dict(sd["cmdB"])

    unit_pi.eval(); vnet.eval(); cmdA.eval(); cmdB.eval()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (host, port)
    dt = 1.0/max(1,fps)

    for ep in range(episodes):
        oA_team, oB_team = env.reset()
        done = False; step = 0
        curr_cmdA = Command.ATTACK; curr_cmdB = Command.DEFEND

        while not done and step < 2000:
            t0 = time.time()
            step += 1

            oA_t = torch.from_numpy(oA_team).float().to(device)
            oB_t = torch.from_numpy(oB_team).float().to(device)

            if (step-1) % K_cmd == 0:
                # greedy command
                logitsA, _ = cmdA(oA_t); cA = torch.argmax(torch.softmax(logitsA, dim=-1)).item()
                logitsB, _ = cmdB(oB_t); cB = torch.argmax(torch.softmax(logitsB, dim=-1)).item()
                curr_cmdA, curr_cmdB = cA, cB

            aA = np.zeros((env.n,), dtype=np.int64)
            aB = np.zeros((env.n,), dtype=np.int64)

            cA_one = onehot_cmd(curr_cmdA)
            cB_one = onehot_cmd(curr_cmdB)

            # greedy per-unit
            for i in range(env.n):
                uA = torch.from_numpy(env.unit_obs("A", i, cA_one)).float().to(device)
                uB = torch.from_numpy(env.unit_obs("B", i, cB_one)).float().to(device)
                aA[i] = int(torch.argmax(torch.softmax(unit_pi(uA), dim=-1)).item())
                aB[i] = int(torch.argmax(torch.softmax(unit_pi(uB), dim=-1)).item())

            oA_team, oB_team, rA, rB, done, info = env.step(aA, aB)

            payload = pack_frame(env, step, info, rA, rB, curr_cmdA, curr_cmdB)
            sock.sendto(json.dumps(payload).encode("utf-8"), addr)

            el = time.time()-t0
            if el < dt: time.sleep(dt-el)

        print(f"[EP {ep+1}] winner={info.get('winner')}  alive A={int(info['aliveA'].sum())} B={int(info['aliveB'].sum())}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None, help="ckpt/hier_XXXX.pt")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7788)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()
    eval_stream(args.model, args.host, args.port, args.fps, args.episodes)
