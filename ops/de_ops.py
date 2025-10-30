# ops/de_ops.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class DEParams:
    F: float = 0.5          # hệ số khuếch đại
    CR: float = 0.9         # crossover rate (binomial)
    p: float = 0.2          # top-p cho pbest (current-to-pbest)
    sigma: float = 0.10     # độ lệch chuẩn cho Gaussian mutation (theo tỉ lệ biên)

def _remove_idx(n: int, i: int) -> np.ndarray:
    """Mảng [0..n) nhưng bỏ i (để chọn rỗng thay)."""
    arr = np.arange(n - 1, dtype=int)
    if i < n - 1:
        arr[i:] += 1
    return arr

def _choice_excl(n: int, k: int, exclude_i: int, rng: np.random.Generator) -> np.ndarray:
    pool = _remove_idx(n, exclude_i)
    return rng.choice(pool, size=k, replace=False)

def _binomial_crossover(target: np.ndarray, mutant: np.ndarray, CR: float, rng: np.random.Generator) -> np.ndarray:
    D = target.shape[-1]
    mask = rng.random(D) < CR
    jrand = int(rng.integers(0, D))     # đảm bảo ít nhất 1 gene lấy từ mutant
    mask[jrand] = True
    return np.where(mask, mutant, target)

def _clip(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    return np.clip(x, lb, ub)

# --- Operator 1: DE/rand/1/bin ---
def de_rand_1_bin(pop: np.ndarray, i: int, rng: np.random.Generator, F: float, CR: float, lb: float, ub: float) -> np.ndarray:
    N, D = pop.shape
    r0, r1, r2 = _choice_excl(N, 3, i, rng)
    mutant = pop[r0] + F * (pop[r1] - pop[r2])
    trial = _binomial_crossover(pop[i], mutant, CR, rng)
    return _clip(trial, lb, ub)

# --- Operator 2: DE/best/2/bin ---
def de_best_2_bin(pop: np.ndarray, i: int, best: np.ndarray, rng: np.random.Generator, F: float, CR: float, lb: float, ub: float) -> np.ndarray:
    N, D = pop.shape
    r1, r2, r3, r4 = _choice_excl(N, 4, i, rng)
    mutant = best + F * (pop[r1] - pop[r2]) + F * (pop[r3] - pop[r4])
    trial = _binomial_crossover(pop[i], mutant, CR, rng)
    return _clip(trial, lb, ub)

# --- Operator 3: DE/current-to-pbest/1 (kiểu JADE) ---
def de_current_to_pbest_1(pop: np.ndarray, i: int, pbest_vec: np.ndarray, rng: np.random.Generator, F: float, CR: float, lb: float, ub: float) -> np.ndarray:
    N, D = pop.shape
    r1, r2 = _choice_excl(N, 2, i, rng)
    mutant = pop[i] + F * (pbest_vec - pop[i]) + F * (pop[r1] - pop[r2])
    trial = _binomial_crossover(pop[i], mutant, CR, rng)
    return _clip(trial, lb, ub)

# --- Operator 4: Gaussian mutation (khai phá) ---
def gaussian_mutation(pop: np.ndarray, i: int, rng: np.random.Generator, sigma: float, CR: float, lb: float, ub: float) -> np.ndarray:
    D = pop.shape[1]
    base = pop[i]
    mask = rng.random(D) < CR
    noise = rng.normal(0.0, sigma, size=D) * (ub - lb)
    trial = np.where(mask, base + noise, base)
    return _clip(trial, lb, ub)

# --- Dispatcher cho env ---
def apply_de_ops(
    pop: np.ndarray,
    fit: np.ndarray,
    op_ids: np.ndarray,
    rng: np.random.Generator,
    lb: float,
    ub: float,
    params: DEParams | None = None
) -> np.ndarray:
    """
    Áp dụng operator tương ứng từng cá thể theo op_ids (0..3).
    """
    if params is None:
        params = DEParams()
    N, D = pop.shape
    new_pop = pop.copy()

    # best và pbest-pool
    best_idx = int(np.argmin(fit))
    best = pop[best_idx]
    k = max(1, int(params.p * N))
    pbest_pool = pop[np.argsort(fit)[:k]]

    for i in range(N):
        op = int(op_ids[i])
        if op == 0:
            trial = de_rand_1_bin(pop, i, rng, params.F, params.CR, lb, ub)
        elif op == 1:
            trial = de_best_2_bin(pop, i, best, rng, params.F, params.CR, lb, ub)
        elif op == 2:
            pbest_vec = pbest_pool[int(rng.integers(0, pbest_pool.shape[0]))]
            trial = de_current_to_pbest_1(pop, i, pbest_vec, rng, params.F, params.CR, lb, ub)
        elif op == 3:
            trial = gaussian_mutation(pop, i, rng, params.sigma, params.CR, lb, ub)
        else:
            trial = pop[i].copy()
        new_pop[i] = trial
    return new_pop
