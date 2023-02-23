import numpy as np
from copy import deepcopy
from utils.model_optimizer import ModelOptimizer


def final_solution_test(optimizer: ModelOptimizer, epsilon, itr):

    dynamics_for_test = deepcopy(optimizer.dynamics)
    v_for_test, main_player = optimizer.model_funcs.init_v_for_test(optimizer.v, epsilon)

    optimizer.calculate_dynamic(v_for_test, dynamics_for_test, calculate_derivative=False)
    last_index = optimizer.model_outer_vars.T - 1
    TotalCost_for_test = optimizer.cost_calculator.calc_total_cost(optimizer.risk_l,
                                                                   dynamics_for_test.S[last_index], v_for_test)

    cost_gap = optimizer.model_funcs.total_cost_for_test(optimizer.cost.TotalCost[itr], TotalCost_for_test,
                                                         main_player)

    return (cost_gap < 0).all()
