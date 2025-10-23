# scripts/smoketest.py
import numpy as np
from envs.aos_gym import AOSGym

def sphere(x): return float(np.sum(x**2))

env = AOSGym(sphere, dim=10, pop_size=50)
obs, info = env.reset(seed=42)

# sanity checks
assert obs.shape == (50, 11), f"obs shape {obs.shape}"
assert np.isfinite(obs).all(), "obs has NaN/Inf"

for t in range(1000):
    action = np.ones(4, dtype=np.float32) / 4.0  # chọn đều 4 operator
    obs, r, done, truncated, info = env.step(action)
    assert np.isfinite(r), f"reward NaN at t={t}"
    assert np.isfinite(obs).all(), f"obs NaN/Inf at t={t}"
    assert obs.min() >= -1.0001 and obs.max() <= 1.0001, "state out of [-1,1]"
    if done: break

print("[PASS] env smoke test 1000 steps")
