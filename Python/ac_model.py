import torch
import torch.nn as nn

class TeamTacticActorCritic(nn.Module):
    """
    팀 전술 정책 π_team(z | team_obs) + 팀 가치 V_team
    """
    def __init__(self, obs_dim, num_tactics, hid=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        self.pi = nn.Linear(hid, num_tactics)
        self.v  = nn.Linear(hid, 1)

    def forward(self, team_obs_b1):  # [B(=1), obs_dim]
        h = self.enc(team_obs_b1)
        logits = self.pi(h)          # [1, Z]
        v = self.v(h).squeeze(-1)    # [1]
        return logits, v


class UnitActorCritic(nn.Module):
    """
    유닛 자율정책 π_unit(a | team_obs, unit_id, z) + 복종도 α + 가치 V
    최종 로짓 = (1-α)*local_logits + α*prior_logits(z)
    prior_logits(z) = tactic_oh @ prior_table
    """
    def __init__(self, obs_dim, num_tactics, n_units, action_dim, hid=256, uid_dim=16):
        super().__init__()
        self.n_units = n_units
        self.num_tactics = num_tactics
        self.action_dim = action_dim

        # 팀 관측 인코더
        self.team_enc = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )

        # 유닛 ID 임베딩
        self.uid_emb = nn.Embedding(n_units, uid_dim)

        in_dim = hid + num_tactics + uid_dim
        self.unit_enc = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )

        # 로컬 정책/가치/복종도 헤드
        self.pi_local = nn.Linear(hid, action_dim)
        self.v_head   = nn.Linear(hid, 1)
        self.alpha    = nn.Sequential(nn.Linear(hid, 1), nn.Sigmoid())

        # 전술별 prior 로짓 테이블(학습 파라미터)
        self.prior = nn.Parameter(torch.zeros(num_tactics, action_dim))
        nn.init.zeros_(self.prior)

    def forward(self, team_obs_b1, tactic_oh_nz, uid_eye_nn):
        """
        team_obs_b1: [1, obs_dim]
        tactic_oh_nz: [n, Z]   (유닛별 전술 원핫)  ← 가끔 [1,n,Z] / [n,1,Z]로 올 수 있어 방어 처리
        uid_eye_nn:   [n, n]   (유닛 ID one-hot)  ← 가끔 [1,n,n]로 올 수 있어 방어 처리
        returns:
          logits:      [n, A]  (최종 혼합 로짓)
          v:           [n]
          alpha:       [n]
          local_logits:[n, A]
          prior_logits:[n, A]
        """
        # --- 안전 차원 평탄화 ---
        if tactic_oh_nz.dim() == 3:
            if tactic_oh_nz.size(0) == 1:
                tactic_oh_nz = tactic_oh_nz.squeeze(0)  # [n,Z]
            elif tactic_oh_nz.size(1) == 1:
                tactic_oh_nz = tactic_oh_nz.squeeze(1)  # [n,Z]
        if uid_eye_nn.dim() == 3:
            if uid_eye_nn.size(0) == 1:
                uid_eye_nn = uid_eye_nn.squeeze(0)      # [n,n]
            elif uid_eye_nn.size(1) == 1:
                uid_eye_nn = uid_eye_nn.squeeze(1)      # [n,n]

        n = tactic_oh_nz.size(0)

        team_emb = self.team_enc(team_obs_b1)     # [1,hid]
        team_emb = team_emb.expand(n, -1)         # [n,hid]

        uid_idx  = uid_eye_nn.argmax(dim=-1)      # [n]
        uid_feat = self.uid_emb(uid_idx)          # [n,uid_dim]

        x = torch.cat([team_emb, tactic_oh_nz, uid_feat], dim=-1)  # [n, hid+Z+uid]
        h = self.unit_enc(x)                                          # [n,hid]

        local_logits = self.pi_local(h)             # [n,A]
        v            = self.v_head(h).squeeze(-1)   # [n]
        alpha        = self.alpha(h).squeeze(-1)    # [n] in (0,1)

        prior_logits = tactic_oh_nz @ self.prior    # [n,A]
        logits = (1.0 - alpha.unsqueeze(-1)) * local_logits + alpha.unsqueeze(-1) * prior_logits
        return logits, v, alpha, local_logits, prior_logits
