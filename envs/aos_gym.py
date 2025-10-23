import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ops.de_ops import apply_de_ops, DEParams

class AOSGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, fobj, dim=10, pop_size=50, bounds=(-100.0, 100.0), fe_budget=None):
        super().__init__()
        self.fobj = fobj
        self.dim = int(dim)
        self.pop_size = int(pop_size)
        self.lb, self.ub = float(bounds[0]), float(bounds[1])
        self.range = max(1e-9, self.ub - self.lb)
        self.fe_budget = int(fe_budget or 10000 * self.dim)

        self.num_ops = 4
        # Action: vector xác suất trên 4 operator
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_ops,), dtype=np.float32)
        # State: [pop_norm, rank_norm_fitness] ∈ [-1, 1]
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(self.pop_size, self.dim + 1), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._prev_best = None
        self._stag = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.fe = 0
        self.pop = self._rng.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        self.fit = self._eval_pop(self.pop)                    # <<< dùng wrapper batch-safe
        self.fe += self.pop_size
        self._prev_best = float(self.fit.min())
        self._stag = 0
        return self._get_state(), {}

    def step(self, action):
        # chuẩn hoá action → phân bố xác suất
        p = np.clip(np.asarray(action, dtype=np.float32).ravel(), 1e-8, 1.0)
        p = p / p.sum()
        op_ids = self._rng.choice(self.num_ops, size=self.pop_size, p=p)

        # áp dụng operator theo từng cá thể (vector hoá vừa đủ)
        new_pop = self._apply_ops(self.pop, op_ids)
        new_fit = self._eval_pop(new_pop)                      # <<< dùng wrapper batch-safe
        self.fe += self.pop_size

        improved = new_fit < self.fit
        self.pop[improved] = new_pop[improved]
        self.fit[improved] = new_fit[improved]

        reward = float(self._calculate_reward())               # hybrid + stagnation-aware
        terminated = self.fe >= self.fe_budget
        truncated = False
        info = {"fe": self.fe, "best": float(self.fit.min())}
        return self._get_state(), reward, terminated, truncated, info

    # ---------- helper: đánh giá fobj an toàn cho cả batch (N,D)->(N,) và per-row (D,)->scalar ----------
    def _eval_pop(self, X: np.ndarray) -> np.ndarray:
        try:
            y = self.fobj(X)                                   # thử batch
            y = np.asarray(y, dtype=float).reshape(-1)
            if y.shape[0] == X.shape[0]:
                return y
        except Exception:
            pass
        # fallback: gọi từng hàng
        return np.apply_along_axis(lambda row: float(self.fobj(row)), 1, X)

    def _get_state(self):
        # chuẩn hoá toạ độ về [-1,1]
        X = (self.pop - self.lb) / self.range * 2.0 - 1.0
        X = np.clip(X, -1.0, 1.0).astype(np.float32)
        # rank-normalize fitness
        ranks = self.fit.argsort().argsort().astype(np.float32)
        ranks /= max(1, self.pop_size - 1)
        return np.concatenate([X, ranks[:, None]], axis=1)

    def _apply_ops(self, pop, op_ids):
        params = DEParams(F=0.5, CR=0.9, p=0.2, sigma=0.10)
        return apply_de_ops(pop, self.fit, op_ids, self._rng, self.lb, self.ub, params)

    # Reward hybrid + stagnation penalty
    def _calculate_reward(self, w1=0.95, w2=0.05, eps_improve=1e-12, k_stag=15, penalty=0.03):
        best = float(self.fit.min())
        delta = (self._prev_best - best) / (abs(self._prev_best) + 1e-9)
        improved = (self._prev_best - best) > eps_improve
        self._prev_best = best

        self._stag = 0 if improved else self._stag + 1
        stagnation_pen = penalty if self._stag >= k_stag else 0.0

        diversity = np.std(self.pop, axis=0).mean() / self.range
        r = w1 * delta + w2 * diversity - stagnation_pen
        return float(np.clip(r, -1.0, 1.0))
