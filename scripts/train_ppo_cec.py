# scripts/train_ppo_cec.py
import argparse
import os
import numpy as np
import torch
import cec2017.functions as fns

from envs.aos_gym import AOSGym
from rl.ppo import PPOAgent, PPOConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--logfile", type=str, default="logs/ppo_cec_f1_10d_run.npz")
    args = parser.parse_args()

    # === CEC function
    getf = getattr(fns, "get_function", None)
    fobj = getf(1) if getf else fns.all_functions[0]

    # === Env
    dim, pop = 10, 50
    fe_budget = pop * 80
    env = AOSGym(fobj, dim=dim, pop_size=pop, fe_budget=fe_budget, bounds=(-100, 100))

    # === Agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PPOConfig(
        steps_per_update=1024, updates=args.updates,
        lr=2e-4, clip_coef=0.1, ent_coef=1e-3, device=device
    )
    agent = PPOAgent(in_features=dim + 1, cfg=cfg)
    num_ops = agent.net.actor.out_features

    # === Schedules
    alpha_hi, alpha_lo = 8.0, 4.0           # tổng alpha: mềm -> cứng
    sharp_lo, sharp_hi = 1.0, 2.5           # power mũ: 1.0 -> 2.5
    agent.net.alpha_bonus_scale = 1.0       # giữ mặc định (có thể tune)

    # === Bandit prior (EMA theo op_improve)
    ema_beta = 0.2
    kappa = 0.8
    eps_bonus = 0.02

    # === Logs
    log_updates, log_losses, log_best = [], [], []
    log_alpha_sum, log_alpha_vec, log_probs = [], [], []
    log_p_used, log_op_hist, log_op_improve = [], [], []

    best = None

    for upd in range(cfg.updates):
        ratio = upd / max(1, (cfg.updates - 1))
        agent.net.dirichlet_alpha_sum = float(alpha_hi - (alpha_hi - alpha_lo) * ratio)
        agent.net.dirichlet_sharp_power = float(sharp_lo + (sharp_hi - sharp_lo) * ratio)

        batch = agent.collect(env, seed=args.seed if upd == 0 else None)
        stats = agent.update(batch)
        best = env.fit.min() if best is None else min(best, env.fit.min())

        # Snapshot & metrics
        with torch.no_grad():
            s = torch.tensor(env._get_state(), dtype=torch.float32, device=agent.cfg.device)
            dist, _ = agent.net.dist_value_dirichlet(s)
            probs = dist.mean.squeeze(0).cpu().numpy()
            alpha_vec = dist.concentration.squeeze(0).cpu().numpy()
            alpha_sum = float(alpha_vec.sum())

        nan_vec = np.full((num_ops,), np.nan, dtype=float)
        li = getattr(env, "_last_info", None) or {}
        p_used = np.array(li.get("p_used", nan_vec), dtype=float)
        op_hist = np.array(li.get("op_hist", nan_vec), dtype=float)
        op_improve = np.array(li.get("op_improve_rate", nan_vec), dtype=float)

        # === Update alpha_bonus theo EMA(op_improve)
        cur_bonus = agent.net.alpha_bonus.detach().cpu().numpy()
        imp = np.nan_to_num(op_improve, nan=0.0, neginf=0.0, posinf=0.0)
        target_bonus = eps_bonus + kappa * imp
        new_bonus = (1.0 - ema_beta) * cur_bonus + ema_beta * target_bonus
        new_bonus = np.clip(new_bonus, 0.0, None)
        agent.net.alpha_bonus.copy_(torch.tensor(new_bonus, dtype=torch.float32, device=agent.cfg.device))

        # === Log
        log_updates.append(upd + 1)
        log_losses.append(float(stats["loss"]))
        log_best.append(float(best))
        log_alpha_sum.append(alpha_sum)
        log_alpha_vec.append(alpha_vec.copy())
        log_probs.append(probs.copy())
        log_p_used.append(p_used.copy())
        log_op_hist.append(op_hist.copy())
        log_op_improve.append(op_improve.copy())

        print(
            f"[upd {upd+1:03d}] loss={stats['loss']:.3f} best={best:.4e} "
            f"probs≈{np.round(probs,3)} alpha≈{np.round(alpha_vec,3)} alpha_sum={alpha_sum:.2f} "
            f"p_used≈{np.round(p_used,3)} op_hist≈{np.round(op_hist,3)} op_improve≈{np.round(op_improve,3)}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.net.state_dict(), "checkpoints/ppo_cec_f1_10d.pt")

    os.makedirs(os.path.dirname(args.logfile), exist_ok=True)
    np.savez(
        args.logfile,
        updates=np.array(log_updates, dtype=int),
        loss=np.array(log_losses, dtype=float),
        best=np.array(log_best, dtype=float),
        alpha_sum=np.array(log_alpha_sum, dtype=float),
        alpha_vec=np.array(log_alpha_vec, dtype=object),
        probs=np.array(log_probs, dtype=object),
        p_used=np.array(log_p_used, dtype=object),
        op_hist=np.array(log_op_hist, dtype=object),
        op_improve=np.array(log_op_improve, dtype=object),
        num_ops=np.int32(num_ops),
    )
    print(f"[DONE] Saved metrics to: {args.logfile}")

if __name__ == "__main__":
    main()
