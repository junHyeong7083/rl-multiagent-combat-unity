"""Boss Raid 환경 패키지

Lost Ark 스타일 1보스 4파티 레이드 환경.
- 플레이어(딜러) 1명 + NPC 3명 (탱커, 힐러, 서포터)
- 8개 패턴, 3페이즈, 어그로/스태거 시스템
"""

from .config import (
    BossConfig, PartyRole, PatternID, PhaseID, BossActionID,
    ROLE_STATS_BOSS,
)
from .boss import Boss
from .patterns import PATTERN_REGISTRY, BasePattern, ActiveTelegraph
from .env import BossRaidEnv
from .fsm_npc import FSMNpcPolicy
from .rewards import RewardComputer

__all__ = [
    "BossConfig", "PartyRole", "PatternID", "PhaseID", "BossActionID",
    "ROLE_STATS_BOSS",
    "Boss", "BossRaidEnv", "FSMNpcPolicy", "RewardComputer",
    "BasePattern", "ActiveTelegraph", "PATTERN_REGISTRY",
]
