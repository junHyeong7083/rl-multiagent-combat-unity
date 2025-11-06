import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, out_dim) if out_dim is not None else None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.out is not None:
            x = self.out(x)
        return x

class TeamPolicy(nn.Module):
    """Discrete tactic z head with value."""
    def __init__(self, obs_dim, num_tactics=4, hidden=256):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden)
        self.pi = nn.Linear(hidden, num_tactics)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.pi(h)
        v = self.v(h).squeeze(-1)
        return logits, v

class UnitActorCritic(nn.Module):
    """Unit policy conditioned on tactic one-hot."""
    def __init__(self, obs_dim, num_tactics, action_dim=10, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + num_tactics, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, z_onehot):
        x = torch.cat([obs, z_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        v = self.v(x).squeeze(-1)
        return logits, v
