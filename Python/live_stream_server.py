
import argparse
import json
import socket
import time
from multiagent_env_5v5 import CombatSelfPlay5v5Env

def run_server(port, fps, max_steps, seed, width, height, n_per_team):
    env = CombatSelfPlay5v5Env(
        width=width, height=height,
        n_per_team=n_per_team, max_steps=max_steps, seed=seed
    )
    env.reset()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", port)
    dt = 1.0 / float(fps)

    print(f"[demo] UDP => {addr}, grid={width}x{height}, n={n_per_team}, max_steps={max_steps}, seed={seed}")
    while True:
        # 랜덤 데모 액션
        aA = env.sample_actions()
        aB = env.sample_actions()
        _, _, rA, rB, done, info = env.step(aA, aB)

        payload = {
            "t": env.t,
            "width": env.width,
            "height": env.height,
            "baseA": info["base_A"],
            "baseB": info["base_B"],
            "A": env.A.tolist(),
            "B": env.B.tolist(),
            "shots": info["shots"],
            "outcome": info["outcome"],
        }
        sock.sendto(json.dumps(payload).encode("utf-8"), addr)

        if done:
            env.reset()
        time.sleep(dt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--n_per_team", type=int, default=30)
    args = ap.parse_args()

    run_server(
        port=args.port, fps=args.fps, max_steps=args.max_steps, seed=args.seed,
        width=args.width, height=args.height, n_per_team=args.n_per_team
    )
