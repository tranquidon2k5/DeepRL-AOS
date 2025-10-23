import os, numpy as np, torch, csv
import cec2017.functions as fns
from envs.aos_gym import AOSGym
from rl.nets import ActorCritic
from rl.ppo import PPOConfig

OPS = ["DE/rand/1/bin","DE/best/2/bin","current-to-pbest/1","Gaussian"]

def get_fobj():
    getf = getattr(fns, "get_function", None)
    return getf(1) if getf else fns.all_functions[0]

def rollout(env, policy=None, steps=9999):
    obs, _ = env.reset(seed=None)
    best = env.fit.min()
    t = 0
    while True and t < steps:
        if policy is None:
            # uniform mixture
            a = np.ones(4, dtype=np.float32) / 4.0
        else:
            s = torch.tensor(obs, dtype=torch.float32)
            dist, _ = policy(s)                        # returns Dirichlet dist
            a = dist.mean.squeeze(0).cpu().numpy()     # dùng mean (deterministic)
        obs, r, done, _, info = env.step(a)
        best = min(best, info["best"]); t += 1
        if done: break
    return best, info["fe"], t

def main():
    dim, pop = 10, 30
    fobj = get_fobj()
    env_u = AOSGym(fobj, dim=dim, pop_size=pop, fe_budget=pop*40, bounds=(-100,100))

    # uniform baseline
    b_u, fe_u, t_u = rollout(env_u, policy=None)

    # RL policy
    net = ActorCritic(in_features=dim+1, num_ops=4)
    ckpt = "checkpoints/ppo_cec_f1_10d.pt"
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy = lambda s: net.dist_value_dirichlet(s)

    env_r = AOSGym(fobj, dim=dim, pop_size=pop, fe_budget=pop*40, bounds=(-100,100))
    b_r, fe_r, t_r = rollout(env_r, policy)

    os.makedirs("reports", exist_ok=True)
    with open("reports/eval_compare_cec1_10d.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","best","fe","steps"])
        w.writerow(["uniform", b_u, fe_u, t_u])
        w.writerow(["ppo-mixture", b_r, fe_r, t_r])
    print("[RESULT] reports/eval_compare_cec1_10d.csv")
    print(f"uniform: best={b_u:.4e}, ppo: best={b_r:.4e}")

if __name__ == "__main__":
    main()
