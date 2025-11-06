import argparse, json, socket, time
import numpy as np
import torch
import torch.nn.functional as F

from multiagent_env_5v5 import CombatSelfPlay5v5Env
from ac_model import ActorCritic

# ----------------------------
# helpers
# ----------------------------
def _to_py(o):
    """NumPy/torch 포함 객체를 JSON 직렬화 가능 형태로 변환"""
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):  # np.int32 등
        return o.item()
    try:
        import torch as _torch
        if isinstance(o, _torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, (_torch.int32, _torch.int64, _torch.float32, _torch.float64)):
            return o.item()
    except Exception:
        pass
    return o

def send_line(conn, obj):
    """한 줄 JSON 전송 (NumPy/torch 안전 변환 포함)"""
    data = (json.dumps(_to_py(obj), ensure_ascii=False) + "\n").encode("utf-8")
    conn.sendall(data)

def load_model_or_none(path, state_dim, action_dim, device):
    if not path:
        return None
    m = ActorCritic(state_dim, action_dim).to(device)
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m

def select_actions(model, obs_np, n_agents, device):
    if model is None:
        return np.random.randint(0, 10, size=n_agents, dtype=np.int32)
    x = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(x)                        # (1, action_dim), (1,1)
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    return np.random.choice(len(probs), size=n_agents, p=probs).astype(np.int32)

# ----------------------------
# server loop
# ----------------------------
def run_server(port, fps, max_steps, seed, ckpt_A, ckpt_B, device,
               episodes, loop_delay, width, height):
    # socket server (고정 1:1 연결)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    print(f"[server] listening on 127.0.0.1:{port}")
    conn, addr = srv.accept()
    print(f"[server] client connected: {addr}")

    delay = 1.0 / max(1, fps)
    epi_idx = 0
    rng = np.random.RandomState(seed)

    try:
        while True:
            # === 에피소드 초기화 ===
            env = CombatSelfPlay5v5Env(
                seed=int(rng.randint(0, 2**31 - 1)),
                max_steps=max_steps,
                width=width, height=height  # ★ 맵 크기
            )
            obs_A, obs_B = env.reset()
            state_dim = obs_A.shape[0]
            action_dim = 10
            n_agents = env.n

            model_A = load_model_or_none(ckpt_A, state_dim, action_dim, device)
            model_B = load_model_or_none(ckpt_B, state_dim, action_dim, device)

            # 메타/리셋 알림
            send_line(conn, {
                "type": "meta", "episode": epi_idx,
                "width": env.width, "height": env.height,
                "n": n_agents, "fps": fps
            })
            send_line(conn, {"type": "reset", "episode": epi_idx})

            # 시뮬 루프
            done = False
            t = 0
            while not done:
                acts_A = select_actions(model_A, obs_A.astype(np.float32), n_agents, device)
                acts_B = select_actions(model_B, obs_B.astype(np.float32), n_agents, device)

                next_A, next_B, rA, rB, done, info = env.step(acts_A, acts_B)

                # 프레임 전송 (shots 포함)
                send_line(conn, {
                    "type": "frame", "episode": epi_idx, "t": t,
                    "A": env.A, "B": env.B,                      # ndarray -> _to_py에서 list로 변환
                    "acts_A": acts_A, "acts_B": acts_B,          # np.array -> 변환됨
                    "rA": rA, "rB": rB,                          # float
                    "shots": info.get("shots", [])               # 발사 궤적
                })

                obs_A, obs_B = next_A, next_B
                t += 1
                time.sleep(delay)

            send_line(conn, {"type": "done", "episode": epi_idx})
            print(f"[server] episode {epi_idx} done")

            epi_idx += 1
            if episodes > 0 and epi_idx >= episodes:
                break
            if loop_delay > 0:
                time.sleep(loop_delay)

    except (BrokenPipeError, ConnectionResetError):
        print("[server] client disconnected")
    finally:
        try:
            conn.close()
        except:
            pass
        srv.close()

# ----------------------------
# entry
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ckpt_A", type=str, default=None)
    ap.add_argument("--ckpt_B", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--episodes", type=int, default=0, help="0=무한 반복, N=에피소드 N번")
    ap.add_argument("--loop_delay", type=float, default=0.2)
    ap.add_argument("--width", type=int, default=24)   # ★ 맵 가로
    ap.add_argument("--height", type=int, default=16)  # ★ 맵 세로
    args = ap.parse_args()

    run_server(args.port, args.fps, args.max_steps, args.seed,
               args.ckpt_A, args.ckpt_B, args.device,
               args.episodes, args.loop_delay,
               args.width, args.height)
