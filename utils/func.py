import numpy as np


def infected(I, v, outer):

    new_I = (outer['beta'] * outer['d']*v*v.T * (1-I)*I.reshape((1, outer['d'].shape[0]))).sum(axis=1)

    return I + new_I


def calculate_derivative(I, outer, v, groups):

    diag = np.diag(np.ones(groups)) + np.ones(groups)
    x_multi = ((1 - I.T) @ I)
    dI = outer['beta'] * outer['d'] * x_multi * v.T * diag
    return dI
