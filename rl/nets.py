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
        h = self.phi(x)                # [B,N,H]
        z = torch.cat([h.mean(1), h.max(1).values], dim=1)  # [B,2H]
        return self.rho(z)             # [B,out]

class ActorCritic(nn.Module):
    def __init__(self, in_features: int, num_ops: int, hidden: int = 128):
        super().__init__()
        self.encoder = DeepSetsEncoder(in_features, hidden, hidden)
        self.actor = nn.Linear(hidden, num_ops)   # dùng cho Dirichlet
        self.critic = nn.Linear(hidden, 1)

    def dist_value_categorical(self, state):
        z = self.encoder(state)
        logits = self.actor(z)
        return Categorical(logits=logits), self.critic(z).squeeze(-1)

    def dist_value_dirichlet(self, state):
        z = self.encoder(state)
        # nồng độ > 0, tránh quá nhọn: softplus + bias nhỏ
        #conc = F.softplus(self.actor(z)) + 0.5      # [B,num_ops]

        tau = 0.15   # <1 làm phân phối sắc hơn
        eps = 0.01   # tránh 0
        conc = F.softplus(self.actor(z)) * tau + eps
        return Dirichlet(conc), self.critic(z).squeeze(-1)
