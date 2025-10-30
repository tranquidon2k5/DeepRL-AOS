from algos.de_baseline import de_baseline
import numpy as np

sphere = lambda x: np.sum(x**2)
best, val = de_baseline(sphere, dim=10, gens=200)
print("Best fitness:", val)
