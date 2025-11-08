import torch
import torch.nn as nn
import torch.nn.functional as F

# 팀 전술(전략) 정책: 팀 관측 -> 전술 logits, 팀가치
class TeamTacticActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_tactics: int):
        super().__init__()
        hid = 256
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, num_tactics)
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, team_obs):  # [B, obs_dim]
        logits = self.pi(team_obs)
        v = self.v(team_obs).squeeze(-1)
        return logits, v


# 유닛 정책: (팀 관측, 전술 one-hot, 유닛ID one-hot) -> 각 유닛의 행동 logits, 가치
class UnitActorCritic(nn.Module):
    def __init__(self, unit_obs_dim: int, num_tactics: int, num_units: int, action_dim: int):
        super().__init__()
        self.num_units = num_units
        self.action_dim = action_dim
        inp = unit_obs_dim + num_tactics + num_units
        hid = 256
        self.net = nn.Sequential(
            nn.Linear(inp, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU()
        )
        self.pi_head = nn.Linear(hid, action_dim)
        self.v_head = nn.Linear(hid, 1)

    # team_obs_rep: [N*B, unit_obs_dim], z_onehot: [N*B, num_tactics], unit_id_oh: [N*B, num_units]
    def forward(self, team_obs_rep, z_onehot, unit_id_oh):
        x = torch.cat([team_obs_rep, z_onehot, unit_id_oh], dim=-1)
        h = self.net(x)
        logits = self.pi_head(h)
        v = self.v_head(h).squeeze(-1)
        return logits, v
