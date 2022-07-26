import numpy as np


def infected(I, v, outer):
    new_I = (outer['beta'] * outer['d'] * v * v.T * (1 - I) * I.reshape((1, outer['d'].shape[0]))).sum(axis=1)

    return I + new_I


def calculate_derivative(I, outer, v, groups):
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
