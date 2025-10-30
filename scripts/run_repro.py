import os, numpy as np
from envs.aos_gym import AOSGym

def read_seeds(path):
    if not os.path.exists(path): return [42,123,2025]
    txt = open(path).read().replace(',', ' ').split()
    return [int(s) for s in txt]

def sphere(x): return float(np.sum(x**2))

def run_one(env):
    steps = 0
    while True:
        action = np.array([0.34, 0.33, 0.29, 0.04], dtype=np.float32)
        _, _, done, _, info = env.step(action)
        steps += 1
        if done: return info["best"], info["fe"], steps

def main():
    os.makedirs("reports", exist_ok=True)
    seeds = read_seeds("scripts/seeds.txt")

    # Sphere-10D
    rows = []
    for s in seeds:
        env = AOSGym(sphere, dim=10, pop_size=60, fe_budget=6000)
        env.reset(seed=s)
        best, fe, steps = run_one(env)
        rows.append((s, best, fe, steps))
    open("reports/repro_sphere10d.csv","w").write(
        "seed,best,fe,steps\n" + "\n".join(f"{s},{b},{fe},{st}" for s,b,fe,st in rows)
    )

    # CEC F1-10D
    try:
        import cec2017.functions as fns
        getf = getattr(fns, "get_function", None)
        fobj = getf(1) if getf else fns.all_functions[0]
        rows = []
        for s in seeds:
            env = AOSGym(fobj, dim=10, pop_size=60, fe_budget=6000, bounds=(-100,100))
            env.reset(seed=s)
            best, fe, steps = run_one(env)
            rows.append((s, best, fe, steps))
        open("reports/repro_cec1_10d.csv","w").write(
            "seed,best,fe,steps\n" + "\n".join(f"{s},{b},{fe},{st}" for s,b,fe,st in rows)
        )
    except Exception as e:
        print("[WARN] CEC not available:", e)

if __name__ == "__main__":
    main()
