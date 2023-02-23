import numpy as np
from utils.model_types import ModelsTypes
from utils.data_classes import CostVariables, ModelOuterVariables, DynamicsVariables
from utils.utils import calc_diag


class ModelFuncs:
    def __init__(self, outer_variables: ModelOuterVariables, model_type: ModelsTypes):
        self.outer_variables = outer_variables
        self.type = model_type

    def init_params(self):
        pass

    def get_grad(self):
        pass

    def calculate_derivative(self):
        pass

    def calc_condition_for_learning_rate_adjust(self):
        pass

    def init_v_for_test(self):
        pass

    def total_cost_for_test(self):
        pass

    def init_common_params(self, max_itr):
        dTotalCost = np.zeros((max_itr, self.outer_variables.groups, 1))
        TotalCost = np.zeros((max_itr, self.outer_variables.groups, 1))

        I = np.zeros((self.outer_variables.T, self.outer_variables.groups, 1))
        S = I.copy()
        return TotalCost, dTotalCost, I, S


class GovModelFuncs(ModelFuncs):
    def __init__(self, outer_variables: ModelOuterVariables, model_type: ModelsTypes):
        super().__init__(outer_variables, model_type)

    def init_params(self, max_itr):
        v = np.zeros(max_itr + 1)
        v[0] = 1
        dS = np.zeros((self.outer_variables.T, self.outer_variables.groups))
        dI = dS.copy()

        return v, dS, dI, *self.init_common_params(max_itr)

    def calculate_grad(self, cost: CostVariables, itr, learning_rate):
        grad = (np.nan_to_num(cost.dTotalCost[itr], posinf=1 / learning_rate,
                              neginf=-1 / learning_rate) * self.outer_variables.populations_proportions).sum()

        return grad

    def calculate_derivative(self, dynamics: DynamicsVariables, v):
        dI = dynamics.dI
        dS = dynamics.dS
        S = dynamics.S
        I = dynamics.I

        new_dS = dS.copy()
        new_dI = dI.copy()
        derivative_delta = np.zeros(dS.shape)
        groups = self.outer_variables.groups
        d = self.outer_variables.d
        beta = self.outer_variables.beta
        for p in range(groups):
            for i in range(groups):
                derivative_delta[p] += beta * d[p, i] * \
                                       (v ** 2 * (dI[i] * (S[p]) + I[i] * dS[p]) + 2 * v * (I[i] * S[p]))[0]

        new_dS = new_dS - derivative_delta
        new_dI = new_dI + derivative_delta - new_dI * self.outer_variables.recovered_rate
        dS_agg = np.expand_dims(new_dS, -1)
        return dS_agg, new_dS, new_dI

    @staticmethod
    def calc_condition_for_learning_rate_adjust(dTotalCost):
        mask = np.ma.masked_invalid(dTotalCost[1:]).any(axis=0)
        return abs((dTotalCost[1:, mask] - dTotalCost[:-1, mask]) / dTotalCost[1:, mask]).mean()

    @staticmethod
    def init_v_for_test(v, epsilon):
        test_epsilon = epsilon * 100
        v_for_test = v.copy()
        v_for_test += test_epsilon if v_for_test <= 0.5 else -test_epsilon

        return v_for_test, None

    @staticmethod
    def total_cost_for_test(TotalCost, TotalCost_test, main_player):
        return TotalCost.sum() - TotalCost_test.sum()


class AnarchyModelFuncs(ModelFuncs):
    def __init__(self, outer_variables: ModelOuterVariables, model_type: ModelsTypes):
        super().__init__(outer_variables, model_type)

    def init_params(self, max_itr):
        v = np.zeros((max_itr + 1, self.outer_variables.groups, 1))
        v[0] = 1
        dS = np.zeros((self.outer_variables.T, self.outer_variables.groups, self.outer_variables.groups))
        dI = dS.copy()

        return v, dS, dI, *self.init_common_params(max_itr)

    def calculate_grad(self, cost: CostVariables, itr, learning_rate):
        grad = cost.dTotalCost[itr]

        return grad

    def calculate_derivative(self, dynamics: DynamicsVariables, v):
        dI = dynamics.dI
        dS = dynamics.dS
        S = dynamics.S
        I = dynamics.I

        new_dS = dS.copy()
        new_dI = dI.copy()
        derivative_delta = np.zeros(dS.shape)

        groups = self.outer_variables.groups
        d = self.outer_variables.d
        beta = self.outer_variables.beta

        for p in range(groups):
            for i in range(groups):
                for w in range(groups):
                    diag, indx = calc_diag(p, i, w)
                    derivative_delta[p, i] += beta * d[p, w] * (v[p] * v[w] * (dI[w, i] * S[p] + I[w] * dS[p, i]) +
                                                                v[indx] * I[w] * S[p] * diag)[0]

        new_dS = new_dS - derivative_delta
        new_dI = new_dI + derivative_delta - new_dI * self.outer_variables.recovered_rate
        dS_agg = np.expand_dims(np.diag(new_dS), -1)
        return dS_agg, new_dS, new_dI

    @staticmethod
    def calc_condition_for_learning_rate_adjust(dTotalCost):
        return abs((dTotalCost[1:] - dTotalCost[:-1]) / dTotalCost[1:])

    @staticmethod
    def init_v_for_test(v, epsilon):
        test_epsilon = epsilon * 100
        v_for_test = v.copy()
        main_player = np.argmax(np.random.rand(*v_for_test.shape))

        v_for_test[main_player] += test_epsilon if v_for_test[main_player] <= 0.5 else -test_epsilon

        return v_for_test, main_player

    @staticmethod
    def total_cost_for_test(TotalCost, TotalCost_test, main_player):
        return (TotalCost - TotalCost_test)[main_player]


def get_model_funcs(model_type: ModelsTypes, outer_variables: ModelOuterVariables):
    factors = {ModelsTypes.anarchy: AnarchyModelFuncs,
               ModelsTypes.gov: GovModelFuncs}
    model_funcs = factors.get(model_type)(outer_variables, model_type)
    if model_funcs:
        return model_funcs
    else:
        raise f'No Model Funcs implemented for {model_type}'

