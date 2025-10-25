# rl/nets.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Dirichlet

class DeepSetsEncoder(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128, out: int = 128):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.rho = nn.Sequential(nn.Linear(hidden * 2, out), nn.ReLU())

    def forward(self, x):  # x: [B,N,F] or [N,F]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        h = self.phi(x)                                  # [B,N,H]
        z = torch.cat([h.mean(1), h.max(1).values], 1)   # [B,2H]
        return self.rho(z)                               # [B,out]

class ActorCritic(nn.Module):
    def __init__(self, in_features: int, num_ops: int, hidden: int = 128):
        super().__init__()
        self.encoder = DeepSetsEncoder(in_features, hidden, hidden)
        self.actor = nn.Linear(hidden, num_ops)
        self.critic = nn.Linear(hidden, 1)

        # Dirichlet hyper
        self.dirichlet_alpha_sum = 6.0          # sẽ anneal từ trainer
        self.min_conc = 0.03
        self.dirichlet_sharp_power = 1.0        # NEW: làm sắc mean
        self.alpha_bonus_scale = 1.0            # NEW: scale cho prior

        # NEW: prior cộng vào alpha, update từ trainer (EMA theo op_improve)
        self.register_buffer("alpha_bonus", torch.zeros(num_ops))

    def dist_value_categorical(self, state):
        z = self.encoder(state)
        logits = self.actor(z)
        return Categorical(logits=logits), self.critic(z).squeeze(-1)

    def dist_value_dirichlet(self, state):
        z = self.encoder(state)

        # 1) head >0 và làm sắc nếu cần
        raw = F.softplus(self.actor(z)) + 1e-6
        pwr = float(self.dirichlet_sharp_power)
        if pwr != 1.0:
            raw = raw.pow(pwr)

        # 2) chuẩn hoá về tổng alpha mục tiêu S
        S = float(self.dirichlet_alpha_sum)
        conc = raw / raw.sum(-1, keepdim=True) * S

        # 3) cộng prior (budgeted) rồi RE-NORM để vẫn giữ tổng = S
        if self.alpha_bonus is not None:
            bonus = self.alpha_bonus.view(1, -1)
            if self.alpha_bonus_scale != 1.0:
                bonus = bonus * float(self.alpha_bonus_scale)
            bonus = bonus.expand_as(conc)
            conc = conc + bonus
            conc = torch.clamp(conc, min=self.min_conc)
            # re-normalize để tổng đúng = S (bonus chỉ đổi hình dạng, không đổi tổng)
            conc = conc / conc.sum(-1, keepdim=True) * S

        return Dirichlet(conc), self.critic(z).squeeze(-1)
