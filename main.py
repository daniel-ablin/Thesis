import numpy as np
from utils.func import infected, calculate_derivative, adam_optimizer_iteration, optimize
from tqdm import tqdm
from math import floor

Recovered_rate = 0
ReSusceptible_rate = 0

learning_rate = 0.01
T = 1000
I0 = 0.01
outer = {'beta': 2.3/30,
         'd': np.array(([0.5, 0.5], [0.5, 0.5])),
         'l': np.array([5, 25]),
         }

v = np.array(outer['d']/5)

groups = outer['d'].shape[0]

rng = 10000
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
m, m_gov = 0, 0
u, u_gov = 0, 0
counter = 0
stop_itr = 50
Threshold = 10 ** -6
test = 10

v, dTotalCost, TotalCost = optimize(T, I0, outer, gov=True, learning_rate=.01, max_itr=10000, epsilon=10**-8, beta_1=.9
                                    , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=10**-6
                                    , seed=None)

v = np.zeros((rng + 1, groups, groups))
v_gov = np.zeros((rng + 1, groups, groups))
TotalFullCost = np.zeros(rng)
TotalFullCost_gov = np.zeros(rng)
dTotalCost = np.zeros((rng, groups, groups))
dTotalCost_gov = np.zeros((rng, groups, groups))
v[0] = np.random.rand()
v_gov[0] = np.random.rand()

pbar = tqdm(range(rng))
for itr in pbar:
    I = np.zeros((T, groups))
    I_gov = np.zeros((T, groups))
    if Recovered_rate > 0:
        R = np.zeros((T, groups))
        R_gov = np.zeros((T, groups))
        R[0, :], R_gov[0, :] = 0, 0
    I[0, :], I_gov[0, :] = I0, I0

    for t in range(T-1):
        I[t+1, :] = infected(I[t, :], v[itr], outer)
        I_gov[t + 1, :] = infected(I_gov[t, :], v_gov[itr], outer)
        if Recovered_rate > 0:
            R[t+1, :] = I[t, :] * Recovered_rate
            I[t + 1, :] -= R[t+1, :]
            R_gov[t + 1, :] = I_gov[t, :] * Recovered_rate
            I_gov[t + 1, :] -= R_gov[t + 1, :]
        if ReSusceptible_rate > 0:
            I[t + 1, :] -= I[t, :] * ReSusceptible_rate
            I_gov[t + 1, :] -= I_gov[t, :] * ReSusceptible_rate

    # InfectoionCost = outer['l'].reshape(groups, 1) * I
    dI = calculate_derivative(I, outer, v[itr], groups)
    dI_gov = calculate_derivative(I_gov, outer, v_gov[itr], groups, gov=True)

    # Cost = 1/v -1
    dCost = -1/v[itr]**2
    dCost_gov = -1/v_gov[itr]**2
    # [[ 9.471823    3.39994729], [52.37598457 23.10380243]]
    TotalCost = outer['l'].reshape(groups, 1) * I[T-1] + 1/v[itr] -1
    TotalCost_gov = outer['l'].reshape(groups, 1) * I_gov[T - 1] + 1 / v_gov[itr] - 1
    # print(TotalCost)
    TotalFullCost[itr] = TotalCost.sum()
    TotalFullCost_gov[itr] = TotalCost_gov.sum()
    dTotalCost[itr] = outer['l'].reshape(groups, 1) * dI + dCost
    dTotalCost_gov[itr] = outer['l'].reshape(groups, 1) * dI_gov + dCost_gov
    decent, m, u = adam_optimizer_iteration(dTotalCost[itr], m, u, beta_1, beta_2, itr, epsilon,
                                            learning_rate/(1+floor(itr/1000)))
    decent_gov, m_gov, u_gov = adam_optimizer_iteration(dTotalCost_gov[itr], m_gov, u_gov, beta_1, beta_2, itr, epsilon,
                                            learning_rate / (1 + floor(itr / 1000)))
    if (abs(dTotalCost[itr]) > epsilon).sum() < 4:
        decent = decent * (abs(dTotalCost[itr]) > epsilon)

    if itr > stop_itr and (abs((dTotalCost[itr-101:itr-1].sum(axis=0) - dTotalCost[itr]*100)) < Threshold).all():
        if ((v[itr] == 1) * (dTotalCost[itr] < 0)).any() or ((v[itr] == 0) * (dTotalCost[itr] > 0)).any():
            print('no close solution')
            break
        elif (abs(dTotalCost[itr]) < epsilon).all():
            print('found solution')
            break
        learning_rate /= 10
        stop_itr = itr + stop_itr
        Threshold /= 100
        # decent = dTotalCost[itr]

    v[itr + 1] = v[itr] - decent  # np.minimum(np.maximum(dTotalCost * learning_rate, -0.01), 0.01)
    v[itr + 1] = np.minimum(np.maximum(v[itr + 1], 0), 1)

    v_gov[itr + 1] = v_gov[itr] - decent_gov.sum()
    v_gov[itr + 1] = np.minimum(np.maximum(v_gov[itr + 1], 0), 1)

    pbar.set_postfix({"dv: ": dTotalCost[itr].sum(),
                      "dv_gov": dTotalCost_gov[itr].sum()})


print('stop')