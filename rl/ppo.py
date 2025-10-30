# rl/ppo.py
from __future__ import annotations
import torch, torch.nn as nn, torch.optim as optim
from dataclasses import dataclass
from typing import Dict
from rl.nets import ActorCritic

@dataclass
class PPOConfig:
    num_ops: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01       # với Dirichlet có thể tăng nhẹ: 0.02–0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    steps_per_update: int = 256
    updates: int = 200
    device: str = "cpu"

class PPOAgent:
    def __init__(self, in_features: int, cfg: PPOConfig):
        self.cfg = cfg
        self.net = ActorCritic(in_features, cfg.num_ops).to(cfg.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg.lr)

    def collect(self, env, seed=None) -> Dict[str, torch.Tensor]:
        self.net.eval()
        obs, _ = env.reset(seed=seed)

        buf = {"state": [], "action": [], "logp": [], "value": [], "reward": [], "done": []}
        for _ in range(self.cfg.steps_per_update):
            s = torch.tensor(obs, dtype=torch.float32, device=self.cfg.device)   # [N,F]
            dist, value = self.net.dist_value_dirichlet(s)                       # Dirichlet over p
            p = dist.sample()                                                    # [1,num_ops]
            logp = dist.log_prob(p)                                              # [1]

            obs2, r, done, _, _ = env.step(p.squeeze(0).detach().cpu().numpy())

            buf["state"].append(s)                           # [N,F]
            buf["action"].append(p)                          # [1,num_ops]
            buf["logp"].append(logp)                         # [1]
            buf["value"].append(value)                       # [1]
            buf["reward"].append(torch.tensor([r], device=self.cfg.device))
            buf["done"].append(torch.tensor([float(done)], device=self.cfg.device))

            obs = obs2
            if done:
                obs, _ = env.reset()

        for k in buf:
            buf[k] = torch.cat(buf[k], dim=0)
        # ép về [T]
        if buf["reward"].dim() == 2: buf["reward"] = buf["reward"].squeeze(-1)
        if buf["done"].dim()   == 2: buf["done"]   = buf["done"].squeeze(-1)
        if buf["value"].dim()  == 2: buf["value"]  = buf["value"].squeeze(-1)
        return buf

    def _gae(self, rewards, values, dones, gamma, lam):
        T = rewards.size(0)
        adv = torch.zeros_like(rewards)
        lastgaelam = 0.0
        next_value = torch.zeros(1, device=values.device)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            lastgaelam = delta + gamma * lam * mask * lastgaelam
            adv[t] = lastgaelam
            next_value = values[t]
        returns = adv + values
        return adv, returns

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.net.train()
        adv, ret = self._gae(batch["reward"], batch["value"], batch["done"],
                             self.cfg.gamma, self.cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        dist, value = self.net.dist_value_dirichlet(batch["state"])  # state: [T,N,F]
        logp = dist.log_prob(batch["action"])                        # action: [T,num_ops] -> [T]
        ratio = torch.exp(logp - batch["logp"])

        pg_loss = -(torch.min(ratio * adv,
                              torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * adv)).mean()
        v_loss = 0.5 * (ret - value).pow(2).mean()
        ent = dist.entropy().mean()
        loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "pg_loss": float(pg_loss.item()),
            "v_loss": float(v_loss.item()),
            "entropy": float(ent.item()),
            "adv_mean": float(adv.mean().item()),
        }
