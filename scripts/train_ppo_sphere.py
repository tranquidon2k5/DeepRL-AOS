# scripts/train_ppo_sphere.py
import argparse
import numpy as np
import torch

from envs.aos_gym import AOSGym
from rl.ppo import PPOAgent, PPOConfig

def sphere(x): return float(np.sum(x**2))

def get_probs(agent, env):
    """Lấy vector xác suất (mean của Dirichlet) ở state hiện tại."""
    with torch.no_grad():
        s = torch.tensor(env._get_state(), dtype=torch.float32, device=agent.cfg.device)
        dist, _ = agent.net.dist_value_dirichlet(s)     # trả về torch.distributions.Dirichlet
        p = dist.mean.squeeze(0).detach().cpu().numpy() # (num_ops,)
    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    dim, pop = 10, 30
    fe_budget = pop * 20  # ~20 thế hệ / episode
    env = AOSGym(sphere, dim=dim, pop_size=pop, fe_budget=fe_budget)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PPOConfig(
        steps_per_update=512,
        updates=args.updates,
        lr=2e-4,
        clip_coef=0.1,
        ent_coef=0.0,
        device=device,
    )
    agent = PPOAgent(in_features=dim + 1, cfg=cfg)

    best = None
    try:
        for upd in range(cfg.updates):
            # seed cho lần đầu để reproducible, các update sau để None cho đa dạng
            batch = agent.collect(env, seed=args.seed if upd == 0 else None)
            stats = agent.update(batch)

            best = env.fit.min() if best is None else min(best, env.fit.min())
            probs = get_probs(agent, env)
            print(
                f"[upd {upd+1:03d}] loss={stats['loss']:.3f} "
                f"entropy={stats['entropy']:.3f} best={best:.4e} "
                f"probs≈{np.round(probs, 3)}"
            )
    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    finally:
        # lưu checkpoint
        import os
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(agent.net.state_dict(), "checkpoints/ppo_sphere_mix.pt")
        print("[SAVED] checkpoints/ppo_sphere_mix.pt")

if __name__ == "__main__":
    main()
