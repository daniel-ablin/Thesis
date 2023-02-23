import numpy as np
from utils.model_types import ModelsTypes

class InitParams:
    def init_params(self):
        pass

    @staticmethod
    def init_common_params(T, groups, max_itr):
        v_gov = np.zeros(max_itr + 1)
        v = np.zeros((max_itr + 1, groups, 1))
        dTotalCost = np.zeros((max_itr, groups, 1))
        TotalCost = np.zeros((max_itr, groups, 1))
        v_gov[0] = 1
        v[0] = 1

        dS = np.zeros((T, groups)) if gov else np.zeros((T, groups, groups))
        dI = dS.copy()

        I = np.zeros((T, groups, 1))
        S = I.copy()
        return v, v_gov, TotalCost, dTotalCost, dS, dI, I, S


class InitGovParams(InitParams):
    def init_params(self, T, groups, max_itr):
        v = np.zeros(max_itr + 1)
        v[0] = 1

        return v, *self.init_common_params(T, groups, max_itr)


class InitOneForAllParams(InitParams):
    def init_params(self, T, groups, max_itr):
        v = np.zeros((max_itr + 1, groups, 1))
        v[0] = 1

        return v, *self.init_common_params(T, groups, max_itr)


def init_params(model_type: ModelsTypes, T, groups, max_itr):
    factors = {ModelsTypes.anarchy: InitOneForAllParams,
               ModelsTypes.gov: InitGovParams}
    return factors[model_type].init_params(T, groups, max_itr)
