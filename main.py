import numpy as np
from utils.func import infected, calculate_derivative

Recovered_rate = 0
ReSusceptible_rate = 0

T = 1000
I0 = 0.01
outer = {'beta': 0.3,
         'd': np.array(([0.5, 0.5], [0.5, 0.5])),
         'lambda': np.array([5, 25])
         }


groups = outer['d'].shape[0]

I = np.zeros((T, groups))
if Recovered_rate > 0:
    R = np.zeros((T, groups))
    R[0, :] = 0
dx = np.zeros(outer['d'].shape)
I[0, :] = I0
v = np.array(outer['d']/5)

for t in range(T-1):
    I[t+1, :] = infected(I[t, :], v, outer)
    if Recovered_rate > 0:
        R[t+1, :] = I[t, :] * Recovered_rate
        I[t + 1, :] -= R[t+1, :]
    if ReSusceptible_rate > 0:
        I[t + 1, :] -= I[t, :] * ReSusceptible_rate

dI = calculate_derivative(I, outer, v, groups)



print('stop')