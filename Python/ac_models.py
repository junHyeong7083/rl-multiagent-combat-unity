import torch
import torch.nn as nn

def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers += [act()]
        elif out_act is not None:
            layers += [out_act()]
    return nn.Sequential(*layers)

class UnitActorCritic(nn.Module):
    """
    유닛 정책 + 팀 가치. 입력 obs_dim_total = env.obs_dim(기본) + 6(커맨더 1-hot)
    """
    def __init__(self, obs_dim_total, n_actions, hidden=256):
        super().__init__()
        self.obs_dim_total = obs_dim_total
        self.n_actions = n_actions
        self.pi = mlp([obs_dim_total, hidden, hidden, n_actions])
        self.v  = mlp([obs_dim_total, hidden, hidden, 1])

    def forward(self, x):
        logits = self.pi(x)
        v = self.v(x).squeeze(-1)
        return logits, v

class CommanderActorCritic(nn.Module):
    """
    커맨더: 팀 관측(기본 obs 평균 등)을 받아 명령 분포/가치를 낸다.
    입력은 여기선 env.obs_dim(기본)로 통일.
    """
    def __init__(self, obs_dim_base, n_cmds, hidden=256):
        super().__init__()
        self.pi = mlp([obs_dim_base, hidden, hidden, n_cmds])
        self.v  = mlp([obs_dim_base, hidden, hidden, 1])

    def forward(self, x):
        logits = self.pi(x)
        v = self.v(x).squeeze(-1)
        return logits, v
