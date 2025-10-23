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
        h = self.phi(x)  # [B,N,H]
        z = torch.cat([h.mean(1), h.max(1).values], dim=1)  # [B,2H]
        return self.rho(z)  # [B,out]

class ActorCritic(nn.Module):
    def __init__(self, in_features: int, num_ops: int, hidden: int = 128,
                 dirichlet_temp: float = 2.0,          # >1 => alpha lớn hơn, sample bớt one-hot
                 dirichlet_alpha_min: float = 0.3,      # sàn cho từng alpha_i
                 dirichlet_alpha_sum: float | None = 6.0):  # tổng alpha mục tiêu (None: không chuẩn hoá)
        super().__init__()
        self.encoder = DeepSetsEncoder(in_features, hidden, hidden)
        self.actor = nn.Linear(hidden, num_ops)
        self.critic = nn.Linear(hidden, 1)

        self.dirichlet_temp = dirichlet_temp
        self.dirichlet_alpha_min = dirichlet_alpha_min
        self.dirichlet_alpha_sum = dirichlet_alpha_sum

    def dist_value_categorical(self, state):
        z = self.encoder(state)
        logits = self.actor(z)
        return Categorical(logits=logits), self.critic(z).squeeze(-1)

    def dist_value_dirichlet(self, state):
        z = self.encoder(state)
        logits = self.actor(z)

        # Chuyển logits -> alpha dương và "điều tiết" độ sắc
        alpha = F.softplus(logits) * self.dirichlet_temp + self.dirichlet_alpha_min  # [B,num_ops]

        # (tuỳ chọn) chuẩn hoá để tổng alpha ổn định -> kiểm soát trực tiếp độ “mềm”
        if self.dirichlet_alpha_sum is not None:
            sum_alpha = alpha.sum(dim=-1, keepdim=True) + 1e-8
            alpha = alpha * (self.dirichlet_alpha_sum / sum_alpha)

        alpha = torch.clamp(alpha, min=1e-3, max=100.0)
        return Dirichlet(alpha), self.critic(z).squeeze(-1)
