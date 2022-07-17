import numpy as np
from utils.func import iteration, calculate_derivative

T = 100
x0 = 0.001
outer = {'beta': 0.05,
         'd': np.array(([0.02, 0.03], [0.05, 0.04])),
         'lambda': np.array([5, 25])
         }


groups = outer['d'].shape[0]

x = np.zeros((T, groups))
dx = np.zeros(outer['d'].shape)
x[0, :] = x0

v = np.array(outer['d']/5)

for t in range(T-1):
    x[t+1, :] = iteration(x[t, :], v, outer)

dx = calculate_derivative(x, outer, groups)



print('stop')