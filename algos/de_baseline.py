import numpy as np

def de_baseline(fobj, dim=10, pop_size=50, gens=1000, F=0.5, CR=0.9):
    lb, ub = -100, 100
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.array([fobj(x) for x in pop])
    best = pop[np.argmin(fit)]
    best_val = fit.min()

    for g in range(gens):
        for i in range(pop_size):
            a,b,c = np.random.choice(pop_size, 3, replace=False)
            mutant = np.clip(pop[a] + F*(pop[b]-pop[c]), lb, ub)
            cross = np.random.rand(dim) < CR
            trial = np.where(cross, mutant, pop[i])
            trial_fit = fobj(trial)
            if trial_fit < fit[i]:
                pop[i], fit[i] = trial, trial_fit
                if trial_fit < best_val:
                    best, best_val = trial, trial_fit
        if g % 100 == 0:
            print(f"Gen {g}: best={best_val:.4e}")
    return best, best_val
