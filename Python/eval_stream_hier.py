import argparse, json, socket, time
import numpy as np
import torch
from multiagent_env_hier import CombatSelfPlayHierEnv
from ac_models import UnitActorCritic, CommanderActorCritic
from commands import N_COMMANDS, onehot_cmd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def send_frame(sock, addr, env, info):
    # Unity 쪽 FrameModels.cs 기준에 맞춰 최대한 보편 필드 구성
    frame = {
        "nexusA": {"x": float(info["nexusA"][0]), "y": float(info["nexusA"][1])},
        "nexusB": {"x": float(info["nexusB"][0]), "y": float(info["nexusB"][1])},
        "unitsA": [{"x": float(env.posA[i,0]), "y": float(env.posA[i,1]), "hp": int(env.hpA[i])} for i in range(env.n)],
        "unitsB": [{"x": float(env.posB[i,0]), "y": float(env.posB[i,1]), "hp": int(env.hpB[i])} for i in range(env.n)],
        "bullets": [{"x": float(b[0]), "y": float(b[1])} for b in (info.get("bullets") or np.zeros((0,2),dtype=np.float32))],
        "shots":   info.get("shots", []),
        "step": int(info.get("step", 0))
    }
    buf = (json.dumps(frame) + "\n").encode("utf-8")
    sock.sendto(buf, addr)

@torch.no_grad()
def greedy_action(model, obs_np):
    t = torch.from_numpy(obs_np).float().to(device)
    logits, _ = model(t)
    p = torch.softmax(logits, dim=-1)
    a = torch.argmax(p, dim=-1).cpu().numpy()
    return a

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelA", required=True)
    ap.add_argument("--modelB", required=True)
    ap.add_argument("--cmdA", required=True)
    ap.add_argument("--cmdB", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--cmd_interval", type=int, default=8)
    args = ap.parse_args()

    env = CombatSelfPlayHierEnv(n_per_team=5, seed=123)
    obs_total = env.obs_dim + N_COMMANDS

    unitA = UnitActorCritic(obs_total, env.n_actions, hidden=256).to(device)
    unitB = UnitActorCritic(obs_total, env.n_actions, hidden=256).to(device)
    cmdA  = CommanderActorCritic(env.obs_dim, N_COMMANDS, hidden=256).to(device)
    cmdB  = CommanderActorCritic(env.obs_dim, N_COMMANDS, hidden=256).to(device)

    unitA.load_state_dict(torch.load(args.modelA, map_location=device))
    unitB.load_state_dict(torch.load(args.modelB, map_location=device))
    cmdA.load_state_dict(torch.load(args.cmdA, map_location=device))
    cmdB.load_state_dict(torch.load(args.cmdB, map_location=device))

    unitA.eval(); unitB.eval(); cmdA.eval(); cmdB.eval()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    dt = 1.0 / max(1, args.fps)

    for ep in range(args.episodes):
        obsA, obsB = env.reset()
        cur_cmdA, cur_cmdB = 0, 1
        done = False; step = 0
        last_send = 0.0

        while not done and step < env.MAX_STEPS:
            # 커맨더 갱신
            if step % args.cmd_interval == 0:
                oA_cmd = obsA[0:1]
                oB_cmd = obsB[0:1]
                cur_cmdA = int(greedy_action(cmdA, oA_cmd)[0])
                cur_cmdB = int(greedy_action(cmdB, oB_cmd)[0])

            ohA = onehot_cmd(cur_cmdA)
            ohB = onehot_cmd(cur_cmdB)

            obsA_17 = np.concatenate([obsA, np.tile(ohA[None,:], (env.n,1))], axis=1)
            obsB_17 = np.concatenate([obsB, np.tile(ohB[None,:], (env.n,1))], axis=1)

            aA = greedy_action(unitA, obsA_17)
            aB = greedy_action(unitB, obsB_17)

            obsA, obsB, rA, rB, done, info = env.step(aA, aB)

            now = time.time()
            if now - last_send >= dt:
                send_frame(sock, addr, env, info)
                last_send = now

            step += 1

        w = info.get("winner", "A")  # 환경에서 반드시 A/B를 보장하지만 안전장치
        print(f"[EP {ep+1}] winner={w}  alive A={(env.hpA>0).sum()} B={(env.hpB>0).sum()}")

if __name__ == "__main__":
    main()
