# live_stream_server.py
# 환경을 직접 실행해 JSON 프레임을 UDP로 송신하는 스트리머 스크립트
# 사용 예:
#   python .\live_stream_server.py --host 127.0.0.1 --port 7788 --fps 10 --width 48 --height 48 --n_per_team 30 --max_steps 240 --seed 0 --obstacles

import argparse
import json
import socket
import sys
import time
import numpy as np

from multiagent_env_5v5 import CombatSelfPlay5v5Env


def main():
    ap = argparse.ArgumentParser()
    # 네트워크/전송
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Unity LiveViewer 수신 호스트")
    ap.add_argument("--port", type=int, default=7788, help="Unity LiveViewer 수신 포트")
    ap.add_argument("--fps", type=int, default=10, help="전송 FPS")

    # 환경 설정
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--n_per_team", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)

    # 옵션
    ap.add_argument("--obstacles", action="store_true", help="장애물/시야차단(LOS) 활성화")
    ap.add_argument("--obstacle_rate", type=float, default=0.06, help="장애물 밀도(0~1)")
    args = ap.parse_args()

    interval = 1.0 / max(1, args.fps)

    # UDP 소켓 준비
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    print(f"[live_stream_server] UDP {addr} / {args.fps} FPS  "
          f"(map {args.width}x{args.height}, n={args.n_per_team}, max_steps={args.max_steps}, seed={args.seed}, obstacles={args.obstacles})",
          file=sys.stderr)

    # 환경 생성
    env = CombatSelfPlay5v5Env(
        width=args.width,
        height=args.height,
        n_per_team=args.n_per_team,
        max_steps=args.max_steps,
        seed=args.seed,
        use_obstacles=bool(args.obstacles),
        obstacle_rate=float(args.obstacle_rate),
    )

    # 초기화
    team_obs_A, team_obs_B = env.reset()
    t = 0

    try:
        while True:
            # 여기서는 랜덤 정책으로 시연 (원하면 학습된 정책 호출로 바꿔도 됨)
            actA = env.sample_actions()
            actB = env.sample_actions()

            _, _, rA, rB, done, info = env.step(actA, actB)

            # Unity가 쓰는 프레임 포맷
            frame = {
                "t": t,
                "width": env.width,
                "height": env.height,
                # 각 유닛 상태: [x,y,hp,fx,fy,cd,base_cd]
                "A": env.A.tolist(),
                "B": env.B.tolist(),
                # 탄막 연출용
                "shots": info.get("shots", []),
                # 장애물 격자(0/1) — 뷰어에서 사용하지 않아도 무방
                "blocks": info.get("blocks", []).tolist() if hasattr(info.get("blocks", []), "tolist") else info.get("blocks", []),
            }

            payload = json.dumps(frame).encode("utf-8")
            sock.sendto(payload, addr)

            t += 1
            if done:
                t = 0
                team_obs_A, team_obs_B = env.reset()

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[live_stream_server] Stopped.", file=sys.stderr)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
