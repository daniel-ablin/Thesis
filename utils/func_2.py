import numpy as np
import pandas as pd
from tqdm import tqdm
from numba import jit, njit
from math import floor
from utils.calculate_derevative import calculate_derivative
from utils.utils import adam_optimizer_iteration
from optimizer_init_params import *


#@njit
def infected(I, S, v, beta, d):
    new_I = (beta * d * v * v.T * S.reshape((d.shape[0], 1)) *
             I.reshape((1, d.shape[0]))).sum(axis=1)

    return np.expand_dims(new_I, -1)

const = -2

def calc_total_cost(l, groups, S, v, elasticity_adjust):
    return l.reshape(groups, 1) * S**const + 1 / v ** elasticity_adjust + elasticity_adjust * v - elasticity_adjust - 1


def calc_dtotal_cost(l, groups, dS_agg, S, dCost):
    return l.reshape(groups, 1) * const * dS_agg * S**(const-1) + dCost


def calc_dcost(v, elasticity_adjust):
    return -elasticity_adjust / (v ** (elasticity_adjust + 1)) + elasticity_adjust


def calc_condition_for_learning_rate_adjust(dTotalCost, gov):
    if gov:
        mask = np.ma.masked_invalid(dTotalCost[1:]).any(axis=0)
        return abs((dTotalCost[1:, mask] - dTotalCost[:-1, mask]) / dTotalCost[1:, mask]).mean()
    else:
        return abs((dTotalCost[1:] - dTotalCost[:-1]) / dTotalCost[1:])


#@njit
def calculate_dynamic(v, T, beta, d, I, S, dS, dI, Recovered_rate, groups=None, gov=None, one_v_for_all=None,
                      without_derivative=False):
    dS_agg = 0
    for t in range(T - 1):
        infected_on_time_t = infected(I[t, :], S[t, :], v, beta, d)
        I[t + 1, :] = np.clip(I[t, :] + infected_on_time_t - I[t, :] * Recovered_rate, 0, 1)
        S[t + 1, :] = np.clip(S[t, :] - infected_on_time_t, 0, 1)
        if not without_derivative:
            dS_agg, dS[t + 1], dI[t + 1] = calculate_derivative(dS[t], dI[t], I[t], S[t], Recovered_rate, beta, d, v,
                                                                groups, gov=gov, one_v_for_all=one_v_for_all)

    return S, I, dS_agg, dS, dI


def update_v(v, grad, learning_rate, sec_smallest_def, min_v):
    decent = grad * learning_rate
    decent = np.minimum(abs(decent), 0.01) * np.sign(decent)

    v_new = v - decent
    if sec_smallest_def:
        min_v = np.partition(v_new.flatten(), 1)[1]
        min_player = np.argmin(v_new)
        grad[min_player] = 0
    v_new = np.minimum(np.maximum(v_new, min_v), 1)

    return v_new

class Optimizer:
    def __init__(self, I0, learning_rate):
        self.I0 = I0
        self.lerning_rate = learning_rate

class GovOptimizer:
    def __init__(self, T, I0, outer, learning_rate=.01, max_itr=10000, epsilon=10**-8,
             Recovered_rate=0, stop_itr=50, threshold=10**-6, sec_smallest_def=False):
        self.outer_params = outer
        self.endo_params = InitGovParams().init_endo_params(max_itr, T, outer.groups)

    def optimize(self):




def optimize(T, I0, outer, gov=False, one_v_for_all=False, learning_rate=.01, max_itr=10000, epsilon=10**-8,
             Recovered_rate=0, stop_itr=50, threshold=10**-6, filter_elasticity=1, sec_smallest_def=False):
    d = outer['d']
    l = outer['l']
    beta = outer['beta']
    min_v = epsilon
    groups = d.shape[0]
    counter = 0

    v, TotalCost, dTotalCost, dS, dI, I, S = init_params(max_itr, T, groups, gov, one_v_for_all)

    elasticity_adjust = 1/filter_elasticity
    msg = 'time out'
    test_results = dict()
    pbar = tqdm(range(max_itr))
    for itr in pbar:
        dS[0, :] = 0
        dI[0, :] = 0
        I[0, :] = I0
        S[0, :] = 1 - I0

        S, I, dS_agg, dS, dI = calculate_dynamic(v[itr], T, beta, d, I, S, dS, dI, Recovered_rate, groups, gov
                                                 , one_v_for_all)

        dCost = calc_dcost(v[itr], elasticity_adjust)

        TotalCost[itr] = calc_total_cost(l, groups, S[T - 1], v[itr], elasticity_adjust)
        dTotalCost[itr] = calc_dtotal_cost(l, groups, dS_agg, S[T-1], dCost)
        if gov:
            grad = np.nan_to_num(dTotalCost[itr], posinf=1/learning_rate, neginf=-1/learning_rate).sum()
        else:
            grad = dTotalCost[itr]

        v[itr + 1] = update_v(v[itr], grad, learning_rate, sec_smallest_def, min_v)

        if not itr==0 and itr%stop_itr == 0:
            if (abs((dTotalCost[itr-stop_itr-1:itr-1].sum(axis=0) - dTotalCost[itr]*stop_itr)) < threshold).all():
                if (abs(grad) < threshold).all():
                    msg = 'found solution'
                    break
                elif ((v[itr] == 1) * (dTotalCost[itr] < 0)).any() or ((v[itr] == epsilon) * (dTotalCost[itr] > 0)).any():
                    msg = 'no close solution'
                    break
        counter_size = 5
        if counter > counter_size:
            cond_nums = calc_condition_for_learning_rate_adjust(dTotalCost[itr-counter_size:itr], gov)
            cond1 = np.where((cond_nums > 0.5).any(axis=0), 2, 1)
            cond2 = np.where((cond_nums < 0.05).all(axis=0), -0.1, 0)
            learning_rate /= cond1 + cond2
            counter = 0
        else:
            counter += 1

        pbar.set_postfix({"dv: ": grad.sum(),
                          "Total_cost": TotalCost[itr].sum(),
                          "Type": gov})

    solution_test=True
    if solution_test and msg=='found solution':
        v_test = v[itr].copy()
        main_player = np.argmax(np.random.rand(*v_test.shape))
        test_epsilon = epsilon*100
        if gov:
            v_test += test_epsilon if v_test <= 0.5 else -test_epsilon
        else:
            v_test[main_player] += test_epsilon if v_test[main_player] <= 0.5 else -test_epsilon
        I_test = I.copy()
        S_test = S.copy()
        I_test[0, :] = I0
        S_test[0, :] = 1 - I0
        S_test, I_test, _, _, _ = calculate_dynamic(v_test, T, beta, d, I_test, S_test, dS, dI, Recovered_rate, without_derivative=True)
        TotalCost_test = calc_total_cost(l, groups, S_test[T - 1], v_test, elasticity_adjust)

        sol = (TotalCost[itr].sum() - TotalCost_test.sum()) if gov else (TotalCost[itr] - TotalCost_test)[main_player]

        test_results['solution'] = (sol < 0).all()

    return {'v': v[itr], 'v_der': dTotalCost[itr], 'cost': TotalCost[itr], 'msg': msg, 'test_results': test_results,
            'S': S[T-1]}


def get_d_matrix(groups):
    base_d = pd.read_csv('d_params.csv', header=None).to_numpy()
    if groups == 2:
        groups = [10]
    if isinstance(groups, list):
        d_row_split = np.split(base_d, groups)
        d_full_split = [np.split(row_split, groups, axis=1) for row_split in d_row_split]
        d = np.array([[split.sum(axis=0).mean() for split in row_split] for row_split in d_full_split]).T
    else:
        d_row_split = np.array_split(base_d, groups)
        d_full_split = [np.array_split(row_split, groups, axis=1) for row_split in d_row_split]
        d = np.array([[split.sum(axis=0).mean() for split in row_split] for row_split in d_full_split]).T

    return d


def optimize_test(T, I0, d, l, beta, gov=False, one_v_for_all=False, learning_rate=.01, max_itr=10000, epsilon=10**-8,
             Recovered_rate=0, stop_itr=50, threshold=10**-6, filter_elasticity=1, sec_smallest_def=False):
    min_v = epsilon
    groups = d.shape[0]
    counter = 0

    v, TotalCost, dTotalCost, dS, dI, I, S = init_params(max_itr, T, groups, gov, one_v_for_all)

    elasticity_adjust = 1/filter_elasticity
    msg = 'time out'
    test_results = dict()
    pbar = tqdm(range(max_itr))
    for itr in pbar:
        dS[0, :] = 0
        dI[0, :] = 0
        I[0, :] = I0
        S[0, :] = 1 - I0

        S, I, dS_agg, dS, dI = calculate_dynamic(v[itr], T, beta, d, I, S, dS, dI, Recovered_rate, groups, gov
                                                 , one_v_for_all)

        dCost = calc_dcost(v[itr], elasticity_adjust)

        TotalCost[itr] = calc_total_cost(l, groups, S[T - 1], v[itr], elasticity_adjust)
        dTotalCost[itr] = calc_dtotal_cost(l, groups, dS_agg, S[T-1], dCost)

        grad = np.nan_to_num(dTotalCost[itr], posinf=0, neginf=0).sum() if gov else dTotalCost[itr]
        decent = grad*learning_rate
        decent = np.minimum(abs(decent), 0.01) * np.sign(decent)

        v[itr + 1] = v[itr] - decent
        if sec_smallest_def:
            min_v = np.partition(v[itr + 1].flatten(), 1)[1]
            min_player = np.argmin(v[itr+1])
            grad[min_player] = 0
        v[itr + 1] = np.minimum(np.maximum(v[itr + 1], min_v), 1)

        if not itr==0 and itr%stop_itr == 0:
            if (abs((dTotalCost[itr-stop_itr-1:itr-1].sum(axis=0) - dTotalCost[itr]*stop_itr)) < threshold).all():
                if (abs(grad) < threshold).all():
                    msg = 'found solution'
                    break
                elif ((v[itr] == 1) * (dTotalCost[itr] < 0)).any() or ((v[itr] == epsilon) * (dTotalCost[itr] > 0)).any():
                    msg = 'no close solution'
                    break

        if counter > 6:
            cond_nums = calc_condition_for_learning_rate_adjust(dTotalCost[itr-6:itr], gov)
            cond1 = np.where((cond_nums > 0.5).any(axis=0), 2, 1)
            cond2 = np.where((cond_nums < 0.05).all(axis=0), -0.1, 0)
            learning_rate /= cond1 + cond2
            counter = 0
        else:
            counter += 1

        pbar.set_postfix({"dv: ": grad.sum(),
                          "Total_cost": TotalCost[itr].sum(),
                          "Type": gov})

    solution_test=True
    if solution_test and msg=='found solution':
        v_test = v[itr].copy()
        main_player = np.argmax(np.random.rand(*v_test.shape))
        test_epsilon = epsilon*100
        if gov:
            v_test += test_epsilon if v_test <= 0.5 else -test_epsilon
        else:
            v_test[main_player] += test_epsilon if v_test[main_player] <= 0.5 else -test_epsilon
        I_test = I.copy()
        S_test = S.copy()
        I_test[0, :] = I0
        S_test[0, :] = 1 - I0
        S_test, I_test, _, _, _ = calculate_dynamic(v_test, T, beta, d, I_test, S_test, dS, dI, Recovered_rate, without_derivative=True)
        TotalCost_test = calc_total_cost(l, groups, S_test[T - 1], v_test, elasticity_adjust)

        sol = (TotalCost[itr].sum() - TotalCost_test.sum()) if gov else (TotalCost[itr] - TotalCost_test)[main_player]

        test_results['solution'] = (sol < 0).all()

    return {'v': v[itr], 'v_der': dTotalCost[itr], 'cost': TotalCost[itr], 'msg': msg, 'test_results': test_results,
            'S': S[T-1]}
