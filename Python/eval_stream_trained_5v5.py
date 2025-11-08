import argparse, json, socket, time, numpy as np, torch
from torch.distributions import Categorical

from multiagent_env_5v5 import CombatSelfPlay5v5Env
from ac_model import TeamTacticActorCritic, UnitActorCriticMultiHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pair(team_ckpt, unit_ckpt, obs_dim, num_tactics, n_units, action_dim=10):
    team = TeamTacticActorCritic(obs_dim, num_tactics).to(DEVICE)
    unit = UnitActorCriticMultiHead(obs_dim, num_tactics, n_units, action_dim).to(DEVICE)
    c1 = torch.load(team_ckpt, map_location=DEVICE)
    c2 = torch.load(unit_ckpt, map_location=DEVICE)
    # 저장 포맷: {"team":..., "unit":...}
    team.load_state_dict(c1["team"] if "team" in c1 else c1["state_dict"], strict=False)
    unit.load_state_dict(c2["unit"] if "unit" in c2 else c2["state_dict"], strict=False)
    team.eval(); unit.eval()
    return team, unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7788)
    ap.add_argument("--fps", type=int, default=10)

    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--n_per_team", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--obstacle_rate", type=float, default=0.06)

    # A/B 각자 ckpt
    ap.add_argument("--A_team_ckpt", type=str, required=True)
    ap.add_argument("--A_unit_ckpt", type=str, required=True)
    ap.add_argument("--B_team_ckpt", type=str, required=True)
    ap.add_argument("--B_unit_ckpt", type=str, required=True)

    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=1.0, help="softmax temperature (>1 더 랜덤)")
    args = ap.parse_args()

    interval = 1.0 / max(1, args.fps)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    print(f"[eval_AB] UDP {addr} / {args.fps}FPS det={args.deterministic} eps={args.eps} temp={args.temp}", flush=True)

    env = CombatSelfPlay5v5Env(width=args.width, height=args.height, n_per_team=args.n_per_team,
                               max_steps=args.max_steps, seed=args.seed,
                               use_obstacles=bool(args.obstacles), obstacle_rate=float(args.obstacle_rate))
    obs_dim = env.get_team_obs_dim(); n_units = env.n; num_tactics = 4; action_dim = 10

    teamA, unitA = load_pair(args.A_team_ckpt, args.A_unit_ckpt, obs_dim, num_tactics, n_units, action_dim)
    teamB, unitB = load_pair(args.B_team_ckpt, args.B_unit_ckpt, obs_dim, num_tactics, n_units, action_dim)

    unit_eye = torch.eye(n_units, device=DEVICE)

    ep = 0
    team_obs_A, team_obs_B = env.reset()
    t = 0

    def sample_team_and_units(team_net, unit_net, team_obs):
        tobs = torch.tensor(team_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits_z, _ = team_net(tobs)
        if args.temp != 1.0: logits_z = logits_z / args.temp
        if args.deterministic:
            z = torch.argmax(logits_z, dim=-1).item()
        else:
            z = Categorical(logits=logits_z).sample().item()

        z_oh = torch.nn.functional.one_hot(torch.tensor([z], device=DEVICE), num_classes=num_tactics).float()
        tobs_rep = tobs.repeat(n_units, 1); z_rep = z_oh.repeat(n_units, 1)
        logits_u, _ = unit_net(tobs_rep, z_rep, unit_eye)
        if args.temp != 1.0: logits_u = logits_u / args.temp
        if args.deterministic:
            a = torch.argmax(logits_u, dim=-1)
        else:
            a = Categorical(logits=logits_u).sample()
        a_np = a.detach().cpu().numpy().astype(np.int32)

        # eps-greedy
        if args.eps > 0.0:
            rnd = np.random.rand(n_units) < args.eps
            a_np[rnd] = np.random.randint(0, action_dim, size=rnd.sum(), dtype=np.int32)
        return a_np

    try:
        while True:
            aA = sample_team_and_units(teamA, unitA, team_obs_A)
            aB = sample_team_and_units(teamB, unitB, team_obs_B)

            nextA, nextB, rA, rB, done, info = env.step(aA, aB)
            team_obs_A, team_obs_B = nextA, nextB

            frame = {
                "ep": ep, "t": t,
                "width": env.width, "height": env.height,
                "A": env.A.tolist(), "B": env.B.tolist(),
                "shots": info.get("shots", []),
                "blocks": info.get("blocks", []).tolist() if hasattr(info.get("blocks", []), "tolist") else info.get("blocks", []),
                "outcome": info.get("outcome"),
            }
            sock.sendto(json.dumps(frame).encode("utf-8"), addr)

            t += 1
            if done:
                ep += 1; t = 0
                team_obs_A, team_obs_B = env.reset()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == "__main__":
    main()
