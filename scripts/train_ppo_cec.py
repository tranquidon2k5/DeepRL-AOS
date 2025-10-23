# scripts/train_ppo_cec.py
import argparse
import numpy as np
import torch
import cec2017.functions as fns

from envs.aos_gym import AOSGym
from rl.ppo import PPOAgent, PPOConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    getf = getattr(fns, "get_function", None)
    fobj = getf(1) if getf else fns.all_functions[0]

    dim, pop = 10, 50
    fe_budget = pop * 80
    env = AOSGym(fobj, dim=dim, pop_size=pop, fe_budget=fe_budget, bounds=(-100, 100))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PPOConfig(steps_per_update=1024, updates=args.updates, lr=2e-4, clip_coef=0.1, ent_coef=0.0, device=device)
    agent = PPOAgent(in_features=dim + 1, cfg=cfg)

    best = None
    for upd in range(cfg.updates):
        batch = agent.collect(env, seed=args.seed if upd == 0 else None)
        stats = agent.update(batch)

        best = env.fit.min() if best is None else min(best, env.fit.min())

        # === PROBS của policy tại state hiện tại ===
        with torch.no_grad():
            s = torch.tensor(env._get_state(), dtype=torch.float32, device=agent.cfg.device)
            dist, _ = agent.net.dist_value_dirichlet(s)
            probs = dist.mean.squeeze(0).detach().cpu().numpy()
            alpha = dist.concentration.squeeze(0).detach().cpu().numpy()
            alpha_sum = float(alpha.sum())
        print(f"... probs≈{np.round(probs,3)} alpha≈{np.round(alpha,3)} alpha_sum={alpha_sum:.2f}")

        # === OP HIST của step cuối trong batch (không step thêm) ===
        p_used = None
        op_hist = None
        if getattr(env, "_last_info", None) is not None:
            if "p" in env._last_info:
                p_used = np.round(env._last_info["p"], 3)
            if "op_hist" in env._last_info:
                op_hist = np.round(env._last_info["op_hist"], 3)

        print(
            f"[upd {upd+1:03d}] loss={stats['loss']:.3f} best={best:.4e} "
            f"probs≈{np.round(probs,3)}"
            + (f" p_used≈{p_used}" if p_used is not None else "")
            + (f" op_hist≈{op_hist}" if op_hist is not None else "")
        )

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.net.state_dict(), "checkpoints/ppo_cec_f1_10d.pt")
    print("[DONE] PPO sanity on CEC-F1-10D")

if __name__ == "__main__":
    main()
