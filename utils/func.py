import numpy as np
import pandas as pd
from tqdm import tqdm
from math import floor
from utils.calculate_derevative import calculate_derivative
from utils.utils import adam_optimizer_iteration


def infected(I, S, v, outer):
    new_I = (outer['beta'] * outer['d'] * v * v.T * S.reshape((outer['d'].shape[0], 1)) *
             I.reshape((1, outer['d'].shape[0]))).sum(axis=1)

    return new_I[:, np.newaxis]


def init_params(max_itr, T, groups, gov, one_v_for_all):
    if one_v_for_all:
        v = np.zeros((max_itr + 1, groups, 1))
        dTotalCost = np.zeros((max_itr, groups, 1))
    elif gov:
        v = np.zeros(max_itr+1)
        dTotalCost = np.zeros((max_itr, groups, 1))
    else:
        v = np.zeros((max_itr + 1, groups, groups))
        dTotalCost = np.zeros((max_itr, groups, groups))
    TotalCost = np.zeros((max_itr, groups, 1))
    v[0] = 0.001 if gov else np.ones((v[0].shape))*0.001

    dS = np.zeros((T, groups)) if gov else np.zeros((T, groups, groups)) if one_v_for_all else np.zeros(
        (T, groups, groups, groups))
    dI = dS.copy()

    I = np.zeros((T, groups, 1))
    S = I.copy()

    return v, TotalCost, dTotalCost, dS, dI, I, S


def optimize(T, I0, outer, gov=False, one_v_for_all=False, learning_rate=.01, max_itr=10000, epsilon=10**-8,
             Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=10**-6, test_epsilon=10**-8,
             derv_test=False, solution_test=False, total_cost_test=False, filter_elasticity=1,
             sec_smallest_def=False):
    min_v = 1
    groups = outer['d'].shape[0]

    v, TotalCost, dTotalCost, dS, dI, I, S = init_params(max_itr, T, groups, gov, one_v_for_all)

    #learning_rate = np.repeat(learning_rate, groups)

    msg = 'time out'
    test_results = dict()
    pbar = tqdm(range(max_itr))
    for itr in pbar:
        dS[0, :] = 0
        dI[0, :] = 0
        I[0, :] = I0
        S[0, :] = 1 - I0
        if derv_test or solution_test or total_cost_test:
            I_test = I[0, :].copy()
            S_test = S[0, :].copy()
            v_test = v[itr].copy()
            main_player_test, sub_player_test = np.random.choice(groups, 2)
            if gov:
                v_test += test_epsilon if v_test <= 0.5 else -test_epsilon
                v_main = v_test
            elif one_v_for_all:
                v_test[main_player_test] += test_epsilon if v_test[main_player_test] <= 0.5 else -test_epsilon
                v_main = v_test[main_player_test]
            else:
                v_test[main_player_test, sub_player_test] += test_epsilon if v_test[main_player_test, sub_player_test] <= 0.5 else -test_epsilon
                v_main = v_test[main_player_test, sub_player_test]
        else:
            test_results = None

        for t in range(T-1):
            infcted_on_time_t = infected(I[t, :], S[t, :], v[itr], outer)
            I[t + 1, :] = I[t, :] + infcted_on_time_t - I[t, :]*Recovered_rate
            S[t + 1, :] = S[t, :] - infcted_on_time_t
            if derv_test or solution_test or total_cost_test:
                infcted_on_time_t_test = infected(I_test, S_test, v_test, outer)
                I_test += infcted_on_time_t_test - I_test * Recovered_rate
                S_test -= infcted_on_time_t_test
            if ReSusceptible_rate > 0:
                I[t + 1, :] -= I[t, :] * ReSusceptible_rate

            dS_agg, dS[t+1], dI[t+1] = calculate_derivative(dS[t], dI[t], I[t], S[t], Recovered_rate, outer, v[itr], groups, gov=gov,
                                                            one_v_for_all=one_v_for_all)


        if derv_test:
            dv_test = -abs(S_test[main_player_test] - S[T - 1][main_player_test]) / test_epsilon

            derv = dS_agg[main_player_test] if gov or one_v_for_all else dS_agg[main_player_test, sub_player_test]

            test_results['derv'] = (abs(derv - dv_test) < 1) and test_results.get('derv', True)

        dCost = -1 / (filter_elasticity*v[itr]) ** (filter_elasticity+1)

        TotalCost[itr] = (outer['l'].reshape(groups, 1) * (1 - S[T - 1]) + 1 / v[itr] - 1)
        dTotalCost[itr] = outer['l'].reshape(groups, 1) * -dS_agg + dCost
        if total_cost_test:
            TotalCost_test = (outer['l'][main_player_test] * (1-S_test[main_player_test]) + 1 / v_main - 1)
            cost = TotalCost[itr][main_player_test] if gov or one_v_for_all else TotalCost[itr][main_player_test, sub_player_test]
            dTotalCost_test = abs(TotalCost_test - cost)/test_epsilon
            cost_derv = dTotalCost[itr][main_player_test] if gov or one_v_for_all else dTotalCost[itr][main_player_test, sub_player_test]

            test_results['cost_derv'] = (abs(cost_derv - dTotalCost_test) < 100) and test_results.get('cost_derv', True)

        grad = dTotalCost[itr].sum() if gov else dTotalCost[itr]
        #decent, m, u = adam_optimizer_iteration(grad, m, u, beta_1, beta_2, itr, epsilon,
        #                                        learning_rate) # / (floor(itr/1000) + 1))
        grad = grad*learning_rate
        decent = np.minimum(abs(grad), 0.01) * np.sign(grad)
        if itr%stop_itr == 0:
            learning_rate /= np.power(10, 1/(100/stop_itr))
            if (abs((dTotalCost[itr-stop_itr-1:itr-1].sum(axis=0) - dTotalCost[itr]*stop_itr)) < threshold).all():
                if (abs(grad) < threshold).all():
                    msg = 'found solution'
                    break
                elif ((v[itr] == 1) * (dTotalCost[itr] < 0)).any() or ((v[itr] == epsilon) * (dTotalCost[itr] > 0)).any():
                    msg = 'no close solution'
                    break


        v[itr + 1] = v[itr] - decent  # np.minimum(np.maximum(dTotalCost * learning_rate, -0.01), 0.01)
        if sec_smallest_def:
            min_v = np.partition(v[itr + 1].flatten(), 1)[1]
        v[itr + 1] = np.minimum(np.maximum(v[itr + 1], epsilon), min_v)

        pbar.set_postfix({"dv: ": grad.sum(),
                          "Total_cost": TotalCost[itr].sum(),
                          "Type": gov})

    if solution_test and msg=='found solution':
        TotalCost_test = (outer['l'].reshape(groups, 1) * (1 - S_test[main_player_test]) + 1 / v_test - 1).sum(axis=1)

        sol = (TotalCost[itr].sum() - TotalCost_test.sum()) if gov else (TotalCost[itr][main_player_test] - TotalCost_test[main_player_test])

        test_results['solution'] = (sol < 0)

    return {'v': v[itr], 'v_der': dTotalCost[itr], 'cost': TotalCost[itr], 'msg': msg, 'test_results': test_results}


def get_d_matrix(groups):
    base_d = pd.read_csv('d_params.csv', header=None).to_numpy()
    if isinstance(groups, list):
        d_row_split = np.split(base_d, groups)
        d_full_split = [np.split(row_split, groups, axis=1) for row_split in d_row_split]
        d = np.array([[split.sum(axis=0).mean() for split in row_split] for row_split in d_full_split]).T
    else:
        d_row_split = np.array_split(base_d, groups)
        d_full_split = [np.array_split(row_split, groups, axis=1) for row_split in d_row_split]
        d = np.array([[split.sum(axis=0).mean() for split in row_split] for row_split in d_full_split]).T

    return d
