"""Layer 1 BT + Layer 2 RL 하이브리드 dispatcher (v3).

흐름:
  1. BT(Layer 1) 가 act() 호출
  2. 반환이 int 이면 그대로 행동 (기믹 인식 대응)
  3. 반환이 None 이면 RL(Layer 2) 로 fall-through

사용 예:
    # boss_streamer.py 에서
    from src.boss.bt_policy import BTPolicy
    from src.boss.hybrid_policy import HybridPolicy
    from src.boss.boss_streamer import RLNpcPolicy  # (예시)

    rl = RLNpcPolicy(net, env, uid, device)
    policy = HybridPolicy(env, uid, rl)
    action = policy.act()
"""
from __future__ import annotations
from typing import Protocol

from .bt_policy import BTPolicy


class _ActLike(Protocol):
    """RL 정책은 act() -> int 인터페이스만 만족하면 됨."""
    def act(self) -> int: ...


class HybridPolicy:
    """2계층 하이브리드: BT → RL fall-through.

    Args:
        env: BossRaidEnv
        uid: 이 정책이 제어하는 유닛 id
        rl_policy: .act() -> int 를 구현한 RL 정책 객체
    """

    def __init__(self, env, uid: int, rl_policy: _ActLike):
        self.env = env
        self.uid = uid
        self.bt = BTPolicy(env, uid)
        self.rl = rl_policy

        # 통계 (BT vs RL 발동 빈도 측정용)
        self.bt_hits = 0
        self.rl_hits = 0

    def act(self) -> int:
        bt_action = self.bt.act()
        if bt_action is not None:
            self.bt_hits += 1
            return int(bt_action)
        self.rl_hits += 1
        return int(self.rl.act())

    def reset_stats(self) -> None:
        self.bt_hits = 0
        self.rl_hits = 0

    def stats(self) -> dict:
        total = self.bt_hits + self.rl_hits
        return {
            "bt_hits": self.bt_hits,
            "rl_hits": self.rl_hits,
            "bt_ratio": self.bt_hits / max(1, total),
        }
