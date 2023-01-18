import numpy as np

class EndofParams:
    def __init__(self, v, TotalCost, dTotalCost, dS, dI, I, S):
        self.v = v
        self.TotalCost = TotalCost
        self.dTotalCost = dTotalCost
        self.dS = dS
        self.dI = dI
        self.I = I
        self.S = S


class InitParams:
    def init_common_params_and_create_endo_class(self, T, groups, v, max_itr, dTotalCost, dS):
        TotalCost = np.zeros((max_itr, groups, 1))
        v[0] = 1

        dI = dS.copy()

        I = np.zeros((T, groups, 1))
        S = I.copy()

        endo_params = EndofParams(v, TotalCost, dTotalCost, dS, dI, I, S)

        return endo_params


class InitGovParams(InitParams):
    def init_endo_params(self, max_itr, T, groups):
        v = np.zeros(max_itr + 1)
        dTotalCost = np.zeros((max_itr, groups, 1))

        dS = np.zeros((T, groups))

        endo_params = self.init_common_params_and_create_endo_class(T, groups, v, max_itr, dTotalCost, dS)

        return endo_params


class InitOneVForAllParams(InitParams):
    def init_endo_params(self, max_itr, T, groups):
        v = np.zeros((max_itr + 1, groups, 1))
        dTotalCost = np.zeros((max_itr, groups, 1))

        dS = np.zeros((T, groups, groups))

        endo_params = self.init_common_params_and_create_endo_class(T, groups, v, max_itr, dTotalCost, dS)

        return endo_params


class InitOneVForEachParams(InitParams):
    def init_endo_params(self, max_itr, T, groups):
        v = np.zeros((max_itr + 1, groups, groups))
        dTotalCost = np.zeros((max_itr, groups, groups))

        dS = np.zeros((T, groups, groups, groups))

        endo_params = self.init_common_params_and_create_endo_class(T, groups, v, max_itr, dTotalCost, dS)

        return endo_params
