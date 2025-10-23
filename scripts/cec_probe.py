import numpy as np
import cec2017  .functions as fns
from envs.aos_gym import AOSGym

getf = getattr(fns, "get_function", None)
fobj = getf(1) if getf else fns.all_functions[0]   # F1

env = AOSGym(fobj, dim=10, pop_size=60, fe_budget=6000, bounds=(-100,100))
obs, _ = env.reset(seed=2025)

while True:
    action = np.array([0.34, 0.33, 0.29, 0.04], dtype=np.float32)
    obs, r, done, _, info = env.step(action)
    if done: break

print("CEC-F1-10D best:", info["best"], "FE:", info["fe"])
print("[PASS] cec_probe")
