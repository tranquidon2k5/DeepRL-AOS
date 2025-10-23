import numpy as np
import wandb                             # <-- thêm
from envs.aos_gym import AOSGym

def sphere(x): return float(np.sum(x**2))

env = AOSGym(sphere, dim=10, pop_size=50, fe_budget=5000)
obs, _ = env.reset(seed=42)

wandb.init(project="DeepRL-AOS", name="env-smoke", mode="offline")   # <-- thêm

for t in range(1000):
    action = np.ones(4, dtype=np.float32) / 4.0
    obs, r, done, _, info = env.step(action)
    wandb.log({"t": t, "reward": r, "best": info["best"]})           # <-- thêm
    if done: break

wandb.finish()                                                        # <-- thêm
print("[PASS] env smoke test 1000 steps")
