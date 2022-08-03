import numpy as np
from tqdm import tqdm
from math import floor


def infected(I, v, outer):
    new_I = (outer['beta'] * outer['d'] * v * v.T * (1 - I) * I.reshape((1, outer['d'].shape[0]))).sum(axis=1)

    return I + new_I


def calculate_derivative(I, outer, v, groups, gov=False):
    if gov:
        diag = np.ones(groups) * 2
    else:
        diag = np.diag(np.ones(groups)) + np.ones(groups)
    x_multi = ((1 - I.T) @ I)
    dI = outer['beta'] * outer['d'] * x_multi * v.T * diag
    return dI


def adam_optimizer_iteration(grad, m, u, beta_1, beta_2, itr, epsilon, learning_rate):
    m = beta_1 * m + (1 - beta_1) * grad
    u = beta_2 * u + (1 - beta_2) * grad ** 2
    m_hat = m / (1 - beta_1 ** (itr + 1))
    u_hat = u / (1 - beta_2 ** (itr + 1))
    adam = (learning_rate * m_hat) / (u_hat ** 0.5 + epsilon)

    return adam, m, u


def optimize(T, I0, outer, gov=False, learning_rate=.01, max_itr=10000, epsilon=10**-8, beta_1=.9, beta_2=.999
             , Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=10**-6, seed=None):
    m = 0
    u = 0
    rand_gen = np.random.default_rng(seed)
    groups = outer['d'].shape[0]
    v = np.zeros((max_itr + 1, groups, groups))
    TotalCost = np.zeros((max_itr, groups))
    dTotalCost = np.zeros((max_itr, groups, groups))
    v[0] = rand_gen.random(1) if gov else rand_gen.random((groups, groups))

    pbar = tqdm(range(max_itr))
    for itr in pbar:
        I = np.zeros((T, groups))
        I_gov = np.zeros((T, groups))
        if Recovered_rate > 0:
            R = np.zeros((T, groups))
            R_gov = np.zeros((T, groups))
            R[0, :], R_gov[0, :] = 0, 0
        I[0, :], I_gov[0, :] = I0, I0

        for t in range(T - 1):
            I[t + 1, :] = infected(I[t, :], v[itr], outer)
            if Recovered_rate > 0:
                R[t + 1, :] = I[t, :] * Recovered_rate
                I[t + 1, :] -= R[t + 1, :]
            if ReSusceptible_rate > 0:
                I[t + 1, :] -= I[t, :] * ReSusceptible_rate

        # InfectoionCost = outer['l'].reshape(groups, 1) * I
        dI = calculate_derivative(I, outer, v[itr], groups)

        # Cost = 1/v -1
        dCost = -1 / v[itr] ** 2
        # [[ 9.471823    3.39994729], [52.37598457 23.10380243]]
        TotalCost[itr] = (outer['l'].reshape(groups, 1) * I[T - 1] + 1 / v[itr] - 1).sum(axis=1)
        # print(TotalCost)
        dTotalCost[itr] = outer['l'].reshape(groups, 1) * dI + dCost
        grad = dTotalCost[itr].sum() if gov else dTotalCost[itr]
        decent, m, u = adam_optimizer_iteration(grad, m, u, beta_1, beta_2, itr, epsilon,
                                                learning_rate / (1 + floor(itr / 1000)))

        if itr%stop_itr == 0:
            if (abs((dTotalCost[itr-stop_itr-1:itr-1].sum(axis=0) - dTotalCost[itr]*stop_itr)) < threshold).all():
                if (abs(grad) < threshold).all():
                    print('found solution')
                    break
                elif ((v[itr] == 1) * (dTotalCost[itr] < 0)).any() or ((v[itr] == 0) * (dTotalCost[itr] > 0)).any():
                    print('no close solution')
                    break

        v[itr + 1] = v[itr] - decent  # np.minimum(np.maximum(dTotalCost * learning_rate, -0.01), 0.01)
        v[itr + 1] = np.minimum(np.maximum(v[itr + 1], 0), 1)

        pbar.set_postfix({"dv: ": dTotalCost[itr].sum(),
                          "Total_cost": TotalCost[itr].sum()})

    return v[:itr], dTotalCost[:itr], TotalCost[:itr]