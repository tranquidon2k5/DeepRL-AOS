import numpy as np
from envs.aos_gym import AOSGym

def sphere(x): return float(np.sum(x**2))

def test_step_shapes():
    env = AOSGym(sphere, dim=10, pop_size=20, fe_budget=2000)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (20, 11)
    a = np.ones(4, dtype=np.float32)/4
    obs, r, done, _, info = env.step(a)
    assert np.isfinite(obs).all() and np.isfinite(r)
