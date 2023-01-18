import numpy as np


class InitParams:
    def __init__(self):
        self.v = None

    def init_common_params(self, dS, T, groups, max_itr):
        dI = dS.copy()
        I = np.zeros((T, groups, 1))
        S = I.copy()
        TotalCost = np.zeros((max_itr, groups, 1))
        self.v[0] = 1
        return dI, I, S, TotalCost


class InitGovParams(InitParams):
    def init_params(self, max_itr, T, groups):
        self.v = np.zeros(max_itr + 1)
        dTotalCost = np.zeros((max_itr, groups, 1))
        dS = np.zeros((T, groups))

        return self.v, dTotalCost, dS, *self.init_common_params(dS, T, groups, max_itr)


class InitOneForAllParams(InitParams):
    def init_params(self, max_itr, T, groups):
        self.v = np.zeros((max_itr + 1, groups, 1))
        dTotalCost = np.zeros((max_itr, groups, 1))
        dS = np.zeros((T, groups, groups))

        return self.v, dTotalCost, dS, *self.init_common_params(dS, T, groups, max_itr)


class InitOneForEachParams(InitParams):
    def init_params(self, max_itr, T, groups):
        self.v = np.zeros((max_itr + 1, groups, groups))
        dTotalCost = np.zeros((max_itr, groups, groups))
        dS = np.zeros((T, groups, groups, groups))

        return self.v, dTotalCost, dS, *self.init_common_params(dS, T, groups, max_itr)