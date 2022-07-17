import numpy as np


def iteration(x, v, outer):
    new_x = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(v.shape[0]):
            new_x[i] = new_x[i] + outer['beta'] * (outer['d'][i, j] - v[i, j] - v[j, i]) * (1 - x[i]) * x[j]

    return x + new_x


def calculate_derivative(x, outer, groups):

    dx = np.zeros(outer['d'].shape)
    diag = np.diag(np.ones(groups)) + np.ones(groups)
    x_multi = (x.T @ x)
    x_sum = x.sum(axis=0)
    for i in range(groups):
        for j in range(groups):
            dx[i, j] = -outer['beta'] * (x_sum[i] * x_multi[i, j])
    return dx * diag
