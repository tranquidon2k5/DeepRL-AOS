# scripts/train_ppo_cec.py
import argparse
import os
import sys
import numpy as np
import torch
import yaml
import cec2017.functions as fns

# Make sure project root is on sys.path (scripts/..)
CURRENT_FILE = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from envs.aos_gym import AOSGym
from rl.ppo import PPOAgent, PPOConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--function_id", type=int, default=1)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--fe_budget", type=int, default=None)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="logs")
    args = parser.parse_args()

    cfg_yaml = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f) or {}

    def get_cfg(key, default):
        return cfg_yaml.get(key, default)

    function_id = int(get_cfg("function_id", args.function_id))
    print(function_id)
    getf = getattr(fns, "get_function", None)
    if getf is not None:
        fobj = getf(function_id)
        
    else:
        print("cec2017.functions has no get_function")
        try:
            fobj = fns.all_functions[function_id - 1]
        except Exception:
            print("cannot get function by id")
            print(len(fns.all_functions))
            fobj = fns.all_functions[0]
    print(fobj.__name__)
    print(fobj(np.zeros((1, args.dim))))

    dim = int(get_cfg("dim", args.dim))
    pop = int(get_cfg("pop_size", args.pop))
    fe_budget = int(args.fe_budget) if args.fe_budget is not None else pop * 500
    env = AOSGym(fobj, dim=dim, pop_size=pop, fe_budget=fe_budget, bounds=(-100, 100))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PPOConfig(
        steps_per_update=1024,
        updates=int(get_cfg("updates", args.updates)),
        lr=float(get_cfg("lr", 2e-4)),
        clip_coef=float(get_cfg("clip_coef", 0.1)),
        ent_coef=float(get_cfg("ent_coef", 1e-3)),
        gamma=float(get_cfg("gamma", 0.99)),
        gae_lambda=float(get_cfg("gae_lambda", 0.95)),
        device=device,
    )
    agent = PPOAgent(in_features=dim + 1, cfg=cfg)
    num_ops = agent.net.actor.out_features

    alpha_hi = float(get_cfg("alpha_hi", 8.0))
    alpha_lo = float(get_cfg("alpha_lo", 4.0))
    sharp_hi = float(get_cfg("sharp_hi", 2.5))
    sharp_lo = float(get_cfg("sharp_lo", 1.0))
    agent.net.alpha_bonus_scale = 1.0

    ema_beta = float(get_cfg("ema_beta", 0.2))
    kappa = float(get_cfg("kappa", 0.8))
    eps_bonus = float(get_cfg("eps_bonus", 0.02))

    log_updates, log_losses, log_best = [], [], []
    log_alpha_sum, log_alpha_vec, log_probs = [], [], []
    log_p_used, log_op_hist, log_op_improve = [], [], []

    seed_value = int(get_cfg("seed", args.seed))
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    run_name = f"ppo_cec_f{function_id}_{dim}d_seed{seed_value}"
    logfile = args.logfile or os.path.join(args.logdir, f"{run_name}.npz")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    print(f"=== START {run_name} | updates={int(get_cfg('updates', args.updates))}, pop={pop}, fe_budget={fe_budget} ===")

    best = None

    for upd in range(cfg.updates):
        ratio = upd / max(1, (cfg.updates - 1))
        agent.net.dirichlet_alpha_sum = float(alpha_hi - (alpha_hi - alpha_lo) * ratio)
        agent.net.dirichlet_sharp_power = float(sharp_lo + (sharp_hi - sharp_lo) * ratio)

        batch = agent.collect(env, seed=seed_value if upd == 0 else None)
        stats = agent.update(batch)
        best = env.fit.min() if best is None else min(best, env.fit.min())

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

        cur_bonus = agent.net.alpha_bonus.detach().cpu().numpy()
        imp = np.nan_to_num(op_improve, nan=0.0, neginf=0.0, posinf=0.0)
        target_bonus = eps_bonus + kappa * imp
        new_bonus = (1.0 - ema_beta) * cur_bonus + ema_beta * target_bonus
        new_bonus = np.clip(new_bonus, 0.0, None)
        agent.net.alpha_bonus.copy_(torch.tensor(new_bonus, dtype=torch.float32, device=agent.cfg.device))

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
            f"probs~{np.round(probs,3)} alpha~{np.round(alpha_vec,3)} alpha_sum={alpha_sum:.2f} "
            f"p_used~{np.round(p_used,3)} op_hist~{np.round(op_hist,3)} op_improve~{np.round(op_improve,3)}"
        )

    np.savez(
        logfile,
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
    print(f"[DONE] Saved metrics to: {logfile}")


if __name__ == "__main__":
    main()
