from data_objects import ComparisonTypes
import numpy as np


class DerivativeCalculator:
    def __init__(self, beta, d, groups, recovered_rate):
        self.beta = beta
        self.d = d
        self.groups = groups
        self.recovered_rate = recovered_rate
        self.type_factory = {ComparisonTypes.gov: self.calculate_derivative_for_v_for_each,
                              ComparisonTypes.anarchy: self.calculate_derivative_for_v_for_each}

    def calculate_derivative_for_v_for_each(self, dS, dI, I, S, v):

        new_dS = dS.copy()
        new_dI = dI.copy()
        derivative_delta = np.zeros(dS.shape)
        groups = self.groups
        d = self.d
        beta = self.beta
        for p in range(groups):
            for i in range(groups):
                for j in range(groups):
                    for w in range(groups):
                        derivative_delta[p, i, j] += beta * (d[p, w] * v[p, w] * v[w, p] *
                                                             (dI[w, i, j] * S[p] + I[w] * dS[p, i, j]))[0]
                    diag = (i == p) + (j == p)
                    indx = i if i != p else j

                    derivative_delta[p, i, j] += beta * d[p, indx] * v[indx, p] * I[indx] * (S[p]) * diag[0]

        new_dS = new_dS - derivative_delta
        new_dI = new_dI + derivative_delta - new_dI * self.recovered_rate
        dS_agg = np.expand_dims(np.diag(new_dS), -1)
        return dS_agg, new_dS, new_dI

    def calculate_derivative_for_gov(self, dS, dI, I, S, v):
        new_dS = dS.copy()
        new_dI = dI.copy()
        derivative_delta = np.zeros(dS.shape)
        groups = self.groups
        d = self.d
        beta = self.beta
        for p in range(groups):
            for i in range(groups):
                derivative_delta[p] += beta * d[p, i] * \
                                       (v ** 2 * (dI[i] * (S[p]) + I[i] * dS[p]) + 2 * v * (I[i] * S[p]))[0]

        new_dS = new_dS - derivative_delta
        new_dI = new_dI + derivative_delta - new_dI * self.recovered_rate
        dS_agg = np.expand_dims(new_dS, -1)
        return dS_agg, new_dS, new_dI

    def calculate_derivative(self, dS, dI, I, S, v, type):
        derivative_func = self.type_factory.get(type)
        if derivative_func:
            return derivative_func(dS, dI, I, S, v)
        else:
            raise 'derivative type not implemented'