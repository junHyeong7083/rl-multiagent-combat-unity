import torch
import torch.nn as nn
import torch.nn.functional as F

class CommanderPolicy(nn.Module):
    """팀 전역 관측 -> 전술 명령 분포 + V(s)"""
    def __init__(self, obs_dim: int, n_cmds: int, hidden: int = 256):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_cmds)
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        logits = self.pi(x)
        v = self.v(x).squeeze(-1)
        return logits, v

class UnitPolicy(nn.Module):
    """유닛 로컬 관측(+지휘 명령 임베딩) -> per-unit 행동 분포"""
    def __init__(self, unit_obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(unit_obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.pi(x)

class TeamValueNet(nn.Module):
    """팀 전역 관측 -> 팀 값함수 V(s)"""
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.v(x).squeeze(-1)
