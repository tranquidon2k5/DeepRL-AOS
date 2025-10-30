#!/usr/bin/env python3
# scripts/run_pso_cec.py
import argparse, os, time, numpy as np

# import PSO từ algos
from algos.pso_baseline import PSOConfig, run_pso

# CEC2017
try:
    import cec2017.functions as cecf
    getf = getattr(cecf, "get_function", None)
    if getf is None:
        ALL = cecf.all_functions
        if ALL is None:
            raise ImportError("cec2017.functions has neither get_function nor all_functions")
        def getf(fid: int):
            return ALL[fid - 1]
except Exception as e:
    raise RuntimeError(f"Cannot import cec2017.functions: {e}")

def main(): 
    ap = argparse.ArgumentParser("PSO baseline for CEC2017")
    ap.add_argument("--function_id", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--pop", type=int, default=1000000)
    ap.add_argument("--lb", type=float, default=-100.0)
    ap.add_argument("--ub", type=float, default=100.0)
    ap.add_argument("--w", type=float, default=0.7298)
    ap.add_argument("--c1", type=float, default=1.49618)
    ap.add_argument("--c2", type=float, default=1.49618)
    ap.add_argument("--vmax_frac", type=float, default=0.2)
    ap.add_argument("--logdir", type=str, default="logs/pso")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    f = getf(args.function_id)
    print(f(np.zeros((1, args.dim))))

    cfg = PSOConfig(
        dim=args.dim, pop=args.pop, updates=args.updates,
        lb=args.lb, ub=args.ub, w=args.w, c1=args.c1, c2=args.c2,
        vmax_frac=args.vmax_frac, seed=args.seed
    )

    t0 = time.time()
    out = run_pso(f, cfg)
    dur = time.time() - t0

    updates = np.arange(1, args.updates + 1, dtype=int)
    best = out["best_hist"]
    loss = best.copy()  # để tương thích tool vẽ của bạn

    meta = {
        "seed": args.seed,
        "function_id": args.function_id,
        "dim": args.dim,
        "pop": args.pop,
        "w": args.w, "c1": args.c1, "c2": args.c2,
        "lb": args.lb, "ub": args.ub,
        "vmax_frac": args.vmax_frac,
        "elapsed_sec": dur,
        "final_best": float(out["gbest_val"]),
    }

    fname = f"cec_f{args.function_id}_d{args.dim}_seed{args.seed}.npz"
    path = os.path.join(args.logdir, fname)
    np.savez_compressed(path, updates=updates, best=best, loss=loss, **meta)
    print(f"[OK] Saved: {path} | final_best={meta['final_best']:.6g} | time={dur:.2f}s")

if __name__ == "__main__":
    main()
