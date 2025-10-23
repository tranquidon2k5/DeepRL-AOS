# scripts/train_ppo_sphere.py
import numpy as np, torch
from envs.aos_gym import AOSGym
from rl.ppo import PPOAgent, PPOConfig

def sphere(x): return float(np.sum(x**2))

def main():
    dim, pop = 10, 50
    # RÚT NGẮN episode: ~40 thế hệ (mỗi step tính fitness cho pop_size cá thể)
    #env = AOSGym(sphere, dim=dim, pop_size=pop, fe_budget=pop * 40)
    env = AOSGym(sphere, dim=10, pop_size=30, fe_budget=30 * 20)  # pop nhỏ hơn, ep ≈ 20 step

    in_features = dim + 1
    #cfg = PPOConfig(steps_per_update=512, updates=200, lr=3e-4, ent_coef=0.02)  # steps↑, ent↑ nhẹ
    cfg = PPOConfig(
        steps_per_update=512,   # ↑
        updates=120,
        lr=2e-4,                 # ↓ nhẹ
        clip_coef=0.1,           # ↓ update “êm” hơn
        ent_coef=0.00           # ↓ entropy bonus để policy dám lệch
    )

    agent = PPOAgent(in_features=in_features, cfg=cfg)

    best = None
    for upd in range(cfg.updates):
        batch = agent.collect(env)
        stats = agent.update(batch)
        best = env.fit.min() if best is None else min(best, env.fit.min())

        # In mixture probs “trung bình” trên vài bước gần nhất (ước lượng từ batch cuối)
        with torch.no_grad():
            from rl.nets import Dirichlet
            # lấy state hiện tại để xem phân phối
            s = torch.tensor(env._get_state(), dtype=torch.float32)
            dist, _ = agent.net.dist_value_dirichlet(s)
            probs = dist.mean.squeeze(0).cpu().numpy() if hasattr(dist, "mean") else dist.concentration.cpu().numpy()
        print(f"[upd {upd+1:03d}] loss={stats['loss']:.3f} entropy={stats['entropy']:.3f} "
              f"best={best:.4e} probs≈{np.round(probs,3)}")

    print("[DONE] PPO sanity on Sphere-10D (mixture)")

if __name__ == "__main__":
    main()
