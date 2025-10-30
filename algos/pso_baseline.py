# algos/pso_baseline.py

from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np

# Dùng bộ đánh giá quần thể an toàn (1D/2D) + cập nhật best-so-far chuẩn
from utils.eval import eval_pop_safe
try:
    from utils.best import update_best
except Exception:
    # fallback nhẹ nếu bạn chưa tạo utils/best.py
    def update_best(best_so_far, fit):
        vals = np.asarray(fit, dtype=float)
        cur = float(np.nanmin(vals))
        return float(min(best_so_far, cur))


@dataclass
class PSOConfig:
    dim: int
    pop: int = 50
    updates: int = 120
    lb: float = -100.0
    ub: float = 100.0
    w: float = 0.7298
    c1: float = 1.49618
    c2: float = 1.49618
    vmax_frac: float = 0.2
    seed: int = 0


def run_pso(f: Callable[[np.ndarray], float], cfg: PSOConfig) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    lb = np.full(cfg.dim, cfg.lb, dtype=float) if np.isscalar(cfg.lb) else np.asarray(cfg.lb, float)
    ub = np.full(cfg.dim, cfg.ub, dtype=float) if np.isscalar(cfg.ub) else np.asarray(cfg.ub, float)
    span = ub - lb
    assert np.all(span > 0), "Bounds must satisfy ub > lb"

    # --- init swarm ---
    X = rng.uniform(lb, ub, size=(cfg.pop, cfg.dim))
    V = np.zeros_like(X)

    # evaluate & init pbest/gbest
    pbest_pos = X.copy()
    pbest_val = eval_pop_safe(f, X)
    g_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    vmax = cfg.vmax_frac * span
    best_hist = np.empty(cfg.updates, dtype=float)
    best_so_far = np.inf  # sẽ cập nhật từ env.fit.min() (ở đây là vals.min())

    for it in range(cfg.updates):
        # --- velocity update ---
        r1 = rng.random(size=(cfg.pop, cfg.dim))
        r2 = rng.random(size=(cfg.pop, cfg.dim))
        V = (cfg.w * V
             + cfg.c1 * r1 * (pbest_pos - X)
             + cfg.c2 * r2 * (gbest_pos - X))
        V = np.clip(V, -vmax, vmax)

        # --- position update + clamp to bounds ---
        X = np.clip(X + V, lb, ub)

        # --- evaluate current population (env.fit) ---
        vals = eval_pop_safe(f, X)  # đây chính là "env.fit" của vòng hiện tại
        # print(vals)
        # --- update personal/global bests (chuẩn PSO) ---
        better = vals < pbest_val
        if np.any(better):
            pbest_pos[better] = X[better]
            pbest_val[better] = vals[better]

        b_idx = int(np.argmin(pbest_val))
        b_val = float(pbest_val[b_idx])
        if b_val < gbest_val:
            gbest_val = b_val
            gbest_pos = pbest_pos[b_idx].copy()

        # --- best giống PPO: best-so-far của env.fit.min() ---
        best_so_far = update_best(best_so_far, vals)
        best_hist[it] = best_so_far

    return {
        "gbest_val": gbest_val,     # min trên pbest (chuẩn PSO)
        "gbest_pos": gbest_pos,
        "best_hist": best_hist,     # best-so-far của env.fit.min() (đồng bộ với PPO)
    }
