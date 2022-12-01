import numpy as np
from scipy.optimize import minimize_scalar


def objective(x):
    return (x - 2) ** 2


res = minimize_scalar(objective, bounds=[-2, -1], method="bounded")
print(res.x, res.fun)
