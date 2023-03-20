from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from utils.cost_calculator import CostCalculator
from utils.data_classes import DynamicsVariables, CostVariables, ModelOuterVariables
from utils.model_types import ModelsTypes
from utils.models_type_funcs import get_model_funcs
from utils.utils import update_v, break_condition_test
from utils.counter import LearningRateCounter


class ModelOptimizer:
    def __init__(self, groups: int, T: int, beta: NDArray[float], d: NDArray[float], risk_l: NDArray[float], model_type: ModelsTypes, recovered_rate=0,
                 I0=1/100000, max_itr=10000, filter_elasticity=1/8, populations_proportions: NDArray[float] = np.array([1])):
        elasticity_adjust = 1 / filter_elasticity
        self.model_outer_vars = ModelOuterVariables(groups, T, beta, recovered_rate, I0, elasticity_adjust, d,
                                                    populations_proportions, risk_l)
        self.model_funcs = get_model_funcs(model_type, self.model_outer_vars)

        v, dS, dI, TotalCost, dTotalCost, I, S = self.model_funcs.init_params(max_itr)
        I[0, :] = self.model_outer_vars.I0
        S[0, :] = 1 - self.model_outer_vars.I0
        self.max_itr = max_itr

        self.v = v
        self.test_results = None

        self.dynamics = DynamicsVariables(I, S, dS, dI)
        self.cost = CostVariables(TotalCost, dTotalCost)

        self.cost_calculator = CostCalculator(self.model_outer_vars)

        self.learning_rate_counter = LearningRateCounter(5)

    def infected(self, I: NDArray[float], S: NDArray[float], v: NDArray[float]) -> NDArray[float]:
        d = self.model_outer_vars.d
        beta = self.model_outer_vars.beta
        new_I = (beta[:, np.newaxis] * d * v * v.T * S.reshape((d.shape[0], 1)) * I.reshape((1, d.shape[0]))).sum(axis=1)

        return np.expand_dims(new_I, -1)

    def calculate_dynamic(self, v: NDArray[float], dynamics: DynamicsVariables, calculate_derivative: bool = True) -> NDArray[float]:
        dS_agg = 0
        I = dynamics.I
        S = dynamics.S
        dS = dynamics.dS
        dI = dynamics.dI
        for t in range(self.model_outer_vars.T - 1):
            infected_on_time_t = self.infected(I[t, :], S[t, :], v)
            I[t + 1, :] = np.clip(I[t, :] + infected_on_time_t - I[t, :] * self.model_outer_vars.recovered_rate, 0, 1)
            S[t + 1, :] = np.clip(S[t, :] - infected_on_time_t, 0, 1)
            if calculate_derivative:
                dS_agg, dS[t + 1], dI[t + 1] = self.model_funcs.calculate_derivative(dynamics[t], v)

        return dS_agg

    def final_solution_test(self, epsilon: float, itr: int) -> bool:

        dynamics_for_test = deepcopy(self.dynamics)
        v_for_test, main_player = self.model_funcs.init_v_for_test(self.v[itr], epsilon)

        self.calculate_dynamic(v_for_test, dynamics_for_test, calculate_derivative=False)
        last_index = self.model_outer_vars.T - 1
        TotalCost_for_test = self.cost_calculator.calc_total_cost(dynamics_for_test.S[last_index], v_for_test)

        cost_gap = self.model_funcs.total_cost_for_test(self.cost.TotalCost[itr], TotalCost_for_test,
                                                        main_player)

        return (cost_gap < 0).all()

    def optimize(self, learning_rate=.01, epsilon=10**-8, stop_itr=50, threshold=10**-6) -> Tuple[bool, str, dict]:
        self.learning_rate_counter.restart_counter()
        final_dynamics_index = self.model_outer_vars.T-1
        msg = 'time out'
        itr = None
        dS_agg = None
        pbar = tqdm(range(self.max_itr))
        for itr in pbar:

            dS_agg = self.calculate_dynamic(self.v[itr], self.dynamics)

            self.cost[itr] = self.cost_calculator.calculate(self.dynamics.S[final_dynamics_index], self.v[itr], dS_agg)

            grad = self.model_funcs.calculate_grad(self.cost, itr, learning_rate)

            self.v[itr+1] = update_v(self.v[itr], grad, learning_rate, epsilon)

            break_condition, msg = break_condition_test(self.cost.dTotalCost, itr, stop_itr, threshold, grad, epsilon,
                                                        self.v[itr])
            if break_condition:
                break
            learning_rate = self.learning_rate_counter.update_learning_rate(self.cost.dTotalCost, itr, learning_rate,
                                                                            self.model_funcs)

            pbar.set_postfix({"dv: ": grad.sum(),
                              "Total_cost": self.cost.TotalCost[itr].sum(),
                              "Type": self.model_funcs.type})

        if msg == 'found solution':
            self.test_results = self.final_solution_test(epsilon, itr)

        sol = dict(v=self.v[itr], v_der=self.cost.dTotalCost[itr],
                   cost=self.cost.TotalCost[itr] * self.model_outer_vars.populations_proportions, msg=msg,
                   test_results=self.test_results, S=self.dynamics.S[final_dynamics_index],
                   I=self.dynamics.I[final_dynamics_index], dS_agg=dS_agg)

        return self.test_results, msg, sol
