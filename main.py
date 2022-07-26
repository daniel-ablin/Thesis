import numpy as np
from utils.func import infected, calculate_derivative, adam_optimizer_iteration
from tqdm import tqdm

Recovered_rate = 0
ReSusceptible_rate = 0

learning_rate = 0.01
T = 1000
I0 = 0.01
outer = {'beta': 0.3,
         'd': np.array(([0.5, 0.5], [0.5, 0.5])),
         'l': np.array([5, 25]),
         }

v = np.array(outer['d']/5)

groups = outer['d'].shape[0]

rng = 10000
v = np.zeros((rng + 1, groups, groups))
dTotalCost = np.zeros((rng, groups, groups))
v[0] = np.random.rand()
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
m = 0
u = 0
counter = 0
stop_itr = 100
Threshold = 10 ** -6

pbar = tqdm(range(rng))
for itr in pbar:
    I = np.zeros((T, groups))
    dx = np.zeros(outer['d'].shape)
    if Recovered_rate > 0:
        R = np.zeros((T, groups))
        R[0, :] = 0
    I[0, :] = I0

    for t in range(T-1):
        I[t+1, :] = infected(I[t, :], v[itr], outer)
        if Recovered_rate > 0:
            R[t+1, :] = I[t, :] * Recovered_rate
            I[t + 1, :] -= R[t+1, :]
        if ReSusceptible_rate > 0:
            I[t + 1, :] -= I[t, :] * ReSusceptible_rate

    # InfectoionCost = outer['l'].reshape(groups, 1) * I
    dI = calculate_derivative(I, outer, v[itr], groups)

    # Cost = 1/v -1
    dCost = -1/v[itr]**2
    # [[ 9.471823    3.39994729], [52.37598457 23.10380243]]
    TotalCost = outer['l'].reshape(groups, 1) * I[T-1] + 1/v[itr] -1
    # print(TotalCost)

    dTotalCost[itr] = outer['l'].reshape(groups, 1) * dI + dCost
    decent, m, u = adam_optimizer_iteration(dTotalCost[itr], m, u, beta_1, beta_2, itr, epsilon, learning_rate)

    decent = decent * (abs(dTotalCost[itr]) > epsilon)

    if itr > stop_itr and ((dTotalCost[itr-101:itr-1].sum(axis=0) - dTotalCost[itr]*100) < Threshold).all():
        learning_rate /= 10
        stop_itr = itr + 100
        Threshold /= 100
        decent = dTotalCost[itr]

    v[itr + 1] = v[itr] - decent  # np.minimum(np.maximum(dTotalCost * learning_rate, -0.01), 0.01)
    v[itr + 1] = np.minimum(np.maximum(v[itr + 1], 0.0001), 0.9999)

    pbar.set_postfix({"dv: " : dTotalCost[itr].sum()})


print('stop')