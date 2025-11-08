import torch
import torch.nn as nn
import torch.nn.functional as F


class TeamTacticActorCritic(nn.Module):
    """
    팀 전체 관측 -> 전술(카테고리) 분포 + 팀 가치
    """
    def __init__(self, obs_dim: int, num_tactics: int = 4, hidden: int = 256):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_tactics),
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        logits = self.pi(x)
        v = self.v(x).squeeze(-1)
        return logits, v


class UnitActorCriticMultiHead(nn.Module):
    """
    팀 관측 + 전술 onehot -> 각 유닛의 정책/가치
    - 유닛마다 별도 head(가중치) 보유: ModuleList
    - forward 입력은 (B*n_units, feat) 형태로 넣되, 내부에서 unit index별로 head 적용
    """
    def __init__(self, unit_obs_dim: int, num_tactics: int, num_units: int, action_dim: int,
                 trunk_hidden: int = 256, head_hidden: int = 128):
        super().__init__()
        self.num_units = num_units
        self.action_dim = action_dim

        # 공통 trunk: (team_obs + tactic_onehot)
        in_dim = unit_obs_dim + num_tactics
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_hidden), nn.ReLU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.ReLU(),
        )

        # 유닛별 head (정책/가치)
        self.pi_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, head_hidden), nn.ReLU(),
                nn.Linear(head_hidden, action_dim),
            ) for _ in range(num_units)
        ])
        self.v_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_hidden, head_hidden), nn.ReLU(),
                nn.Linear(head_hidden, 1),
            ) for _ in range(num_units)
        ])

    def forward(self, team_obs_rep, tactic_onehot_rep, unit_ids_onehot):
        """
        team_obs_rep: (B*n, unit_obs_dim)  - 팀 관측을 유닛 수만큼 반복
        tactic_onehot_rep: (B*n, num_tactics)
        unit_ids_onehot: (B*n, n) - onehot, argmax로 유닛 인덱스 알 수 있음
        return: logits (B*n, action_dim), values (B*n,)
        """
        x = torch.cat([team_obs_rep, tactic_onehot_rep], dim=-1)  # (B*n, in_dim)
        h = self.trunk(x)
        # 유닛 인덱스
        idx = unit_ids_onehot.argmax(dim=-1)  # (B*n,)

        # 각 행별로 해당 유닛 head 적용
        logits_list = []
        values_list = []
        for k in range(self.num_units):
            mask = (idx == k)
            if mask.any():
                hk = h[mask]
                logits_k = self.pi_heads[k](hk)
                v_k = self.v_heads[k](hk).squeeze(-1)
                logits_list.append((mask, logits_k))
                values_list.append((mask, v_k))

        # 결과 재조합
        logits = torch.zeros(h.size(0), self.action_dim, device=h.device, dtype=h.dtype)
        values = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
        for mask, part in logits_list:
            logits[mask] = part
        for mask, part in values_list:
            values[mask] = part
        return logits, values
