import argparse, json, socket, time, torch
import numpy as np
from torch.distributions import Categorical

from multiagent_env_5v5 import CombatSelfPlay5v5Env, ACTION_NUM
from ac_model import TeamTacticActorCritic, UnitActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_state_safely(model, blob, prefer_key: str):
    if isinstance(blob, dict):
        if prefer_key in blob:           state = blob[prefer_key]
        elif "state_dict" in blob:       state = blob["state_dict"]
        elif "model_state_dict" in blob: state = blob["model_state_dict"]
        else:                            state = blob
    else:
        state = blob
    model.load_state_dict(state, strict=False)

def load_pair(team_ckpt, unit_ckpt, obs_dim, num_tactics, n_units, action_dim):
    team = TeamTacticActorCritic(obs_dim, num_tactics).to(DEVICE)
    unit = UnitActorCritic(obs_dim, num_tactics, n_units, action_dim).to(DEVICE)
    t_blob = torch.load(team_ckpt, map_location=DEVICE)
    u_blob = torch.load(unit_ckpt, map_location=DEVICE)
    _load_state_safely(team, t_blob, prefer_key="team")
    _load_state_safely(unit, u_blob, prefer_key="unit")
    team.eval(); unit.eval()
    return team, unit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teamA', type=str, required=True)
    ap.add_argument('--unitA', type=str, required=True)
    ap.add_argument('--teamB', type=str, required=True)
    ap.add_argument('--unitB', type=str, required=True)
    ap.add_argument('--host', type=str, default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7788)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--temp', type=float, default=1.25)
    ap.add_argument('--eps', type=float, default=0.0)
    ap.add_argument('--print-every', type=int, default=0)
    args = ap.parse_args()

    env = CombatSelfPlay5v5Env()
    obs_dim = env.get_team_obs_dim(); n_units = env.n; action_dim = ACTION_NUM
    num_tactics = 4

    teamA, unitA = load_pair(args.teamA, args.unitA, obs_dim, num_tactics, n_units, action_dim)
    teamB, unitB = load_pair(args.teamB, args.unitB, obs_dim, num_tactics, n_units, action_dim)

    with torch.no_grad():
        a_norm = sum(p.detach().abs().mean().item() for p in unitA.parameters())
        b_norm = sum(p.detach().abs().mean().item() for p in unitB.parameters())
    print(f"[param mean] unitA={a_norm:.6f}, unitB={b_norm:.6f}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    obsA, obsB = env.reset()
    dt = 1.0 / max(1, args.fps)
    uid = torch.eye(n_units, device=DEVICE)

    tick = 0
    while True:
        tA = torch.tensor(obsA, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        tB = torch.tensor(obsB, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # 팀 전술 → 유닛별 전술로 확장
        logits_zA, _ = teamA(tA)
        logits_zB, _ = teamB(tB)
        distA = Categorical(logits=logits_zA / args.temp)
        distB = Categorical(logits=logits_zB / args.temp)
        zA_vec = distA.sample((n_units,)).squeeze(-1)  # [n]
        zB_vec = distB.sample((n_units,)).squeeze(-1)  # [n]
        zA_oh  = torch.nn.functional.one_hot(zA_vec, num_classes=num_tactics).float()  # [n,Z]
        zB_oh  = torch.nn.functional.one_hot(zB_vec, num_classes=num_tactics).float()  # [n,Z]

        # 유닛 정책
        logits_uA, _vA, _alphaA, _localA, _priorA = unitA(tA, zA_oh, uid)
        logits_uB, _vB, _alphaB, _localB, _priorB = unitB(tB, zB_oh, uid)

        pa = Categorical(logits=logits_uA / args.temp)
        pb = Categorical(logits=logits_uB / args.temp)
        aA = pa.sample().cpu().numpy()
        aB = pb.sample().cpu().numpy()

        if args.eps > 0.0:
            randA = np.random.rand(n_units) < args.eps
            randB = np.random.rand(n_units) < args.eps
            if randA.any(): aA[randA] = np.random.randint(0, ACTION_NUM, size=randA.sum())
            if randB.any(): aB[randB] = np.random.randint(0, ACTION_NUM, size=randB.sum())

        obsA, obsB, rA, rB, done, info = env.step(aA, aB)

        # === 뷰어 페이로드 ===
        baseA = {"x": int(env.base_A[0]), "y": int(env.base_A[1])} if hasattr(env, "base_A") else None
        baseB = {"x": int(env.base_B[0]), "y": int(env.base_B[1])} if hasattr(env, "base_B") else None

        shots_out = []
        for s in info.get("shots", []):
            sd = s.__dict__ if hasattr(s, "__dict__") else (s if isinstance(s, dict) else None)
            if sd is None: continue
            fx = sd.get("from_x", sd.get("fx", None)); fy = sd.get("from_y", sd.get("fy", None))
            tx = sd.get("to_x",   sd.get("tx", None)); ty = sd.get("to_y",   sd.get("ty", None))
            team = sd.get("team", sd.get("side", None)); hit = bool(sd.get("hit", sd.get("onhit", False)))
            if None not in (fx,fy,tx,ty):
                shots_out.append({"from":[int(fx),int(fy)], "to":[int(tx),int(ty)],
                                  "from_xy":[int(fx),int(fy)], "to_xy":[int(tx),int(ty)],
                                  "team":team, "hit":hit})
            elif "from_xy" in sd and "to_xy" in sd:
                shots_out.append({"from":[int(sd["from_xy"][0]),int(sd["from_xy"][1])],
                                  "to":[int(sd["to_xy"][0]),int(sd["to_xy"][1])],
                                  "from_xy":[int(sd["from_xy"][0]),int(sd["from_xy"][1])],
                                  "to_xy":[int(sd["to_xy"][0]),int(sd["to_xy"][1])],
                                  "team":team, "hit":hit})

        payload = {
            "t": int(info.get("t", 0)),
            "width": int(env.width), "height": int(env.height),
            "map": {"cell": float(getattr(env, "cell_size", 1.0))},
            "rA": float(rA), "rB": float(rB),
            "A": np.asarray(env.A, dtype=np.int32).tolist(),
            "B": np.asarray(env.B, dtype=np.int32).tolist(),
            "baseA": baseA, "baseB": baseB,
            "obstacles": env.export_obstacles() if hasattr(env, "export_obstacles") else [],
            "obs_ver": int(getattr(env, "obs_ver", 0)),
            "shots": shots_out,
            "done": bool(done),
            "outcome": info.get("outcome", None)
        }
        try:
            sock.sendto(json.dumps(payload).encode('utf-8'), addr)
        except Exception as e:
            print(f"[WARN] UDP send failed: {e}")

        tick += 1
        if args.print_every and (tick % args.print_every == 0):
            print(f"[t={info.get('t',0)}] A actions {aA.tolist()} | B actions {aB.tolist()}")

        if done:
            obsA, obsB = env.reset()

        time.sleep(dt)

if __name__ == "__main__":
    main()
