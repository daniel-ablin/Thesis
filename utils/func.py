import numpy as np
from tqdm import tqdm
from math import floor
from utils.calculate_derevative import calculate_derivative
from utils.utils import adam_optimizer_iteration


def infected(I, v, outer):
    new_I = (outer['beta'] * outer['d'] * v * v.T * (1 - I).reshape((outer['d'].shape[0], 1)) *
             I.reshape((1, outer['d'].shape[0]))).sum(axis=1)

    return I + new_I[:, np.newaxis]


def init_params(max_itr, groups, gov, one_v_for_all, seed):
    rand_gen = np.random.default_rng(seed)
    if one_v_for_all:
        v = np.zeros((max_itr + 1, groups, 1))
        dTotalCost = np.zeros((max_itr, groups, 1))
    else:
        v = np.zeros((max_itr + 1, groups, groups))
        dTotalCost = np.zeros((max_itr, groups, groups))
    TotalCost = np.zeros((max_itr, groups))
    v[0] = rand_gen.random(1) if gov else rand_gen.random(v[0].shape)

    return v, TotalCost, dTotalCost


def optimize(T, I0, outer, gov=False, one_v_for_all=False, learning_rate=.01, max_itr=10000, epsilon=10**-8, beta_1=.9, beta_2=.999
             , Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=10**-6, seed=None
             , derv_test=False, solution_test=False, total_cost_test=False):
    m = 0
    u = 0
    groups = outer['d'].shape[0]

    v, TotalCost, dTotalCost = init_params(max_itr, groups, gov, one_v_for_all, seed)

    msg = 'time out'
    pbar = tqdm(range(max_itr))
    for itr in pbar:
        dI = np.zeros(groups) if gov else np.zeros((groups, groups)) if one_v_for_all else np.zeros((groups, groups, groups))
        I = np.zeros((T, groups, 1))
        I[0, :] = I0

        if derv_test or solution_test or total_cost_test:
            test_results = dict()
            I_test = I[0, :].copy()
            v_test = v[itr].copy()
            main_player_test, sub_player_test = np.random.choice(groups, 2)
            if gov:
                v_test += epsilon
            elif one_v_for_all:
                v_test[main_player_test] += epsilon
            else:
                v_test[main_player_test, sub_player_test] += epsilon
        else:
            test_results = None

        if Recovered_rate > 0:
            R = np.zeros((T, groups))
            R[0, :] = 0

        for t in range(T-1):
            I[t + 1, :] = infected(I[t, :], v[itr], outer)
            if derv_test or solution_test or total_cost_test:
                I_test = infected(I_test, v_test, outer)
            if Recovered_rate > 0:
                R[t + 1, :] = I[t, :] * Recovered_rate
                I[t + 1, :] -= R[t + 1, :]
            if ReSusceptible_rate > 0:
                I[t + 1, :] -= I[t, :] * ReSusceptible_rate

            dI_agg, dI = calculate_derivative(dI, I[t], outer, v[itr], groups, gov=gov, one_v_for_all=one_v_for_all)


        if derv_test:
            dv_test = (I_test[main_player_test] - I[T - 1][main_player_test]) / (epsilon)

            derv = dI_agg[main_player_test] if gov or one_v_for_all else dI_agg[main_player_test, sub_player_test]

            test_results['derv'] = (abs(derv - dv_test) < epsilon*100) and test_results.get('derv', True)

        dCost = -1 / v[itr] ** 2

        TotalCost[itr] = (outer['l'].reshape(groups, 1) * I[T - 1] + 1 / v[itr] - 1).sum(axis=1)
        dTotalCost[itr] = outer['l'].reshape(groups, 1) * dI_agg + dCost
        if total_cost_test:
            TotalCost_test = (outer['l'][main_player_test] * I_test[main_player_test] + 1 / v_test[main_player_test] - 1)
            cost = TotalCost[itr][main_player_test] if gov or one_v_for_all else TotalCost[itr][main_player_test, sub_player_test]
            dTotalCost_test = (TotalCost_test - cost)/epsilon
            cost_derv = dTotalCost[itr][main_player_test] if gov or one_v_for_all else dTotalCost[itr][main_player_test, sub_player_test]

            test_results['cost_derv'] = (abs(cost_derv - dTotalCost_test) < epsilon * 1000) and test_results.get('cost_derv', True)

        grad = dTotalCost[itr].sum() if gov else dTotalCost[itr]
        decent, m, u = adam_optimizer_iteration(grad, m, u, beta_1, beta_2, itr, epsilon,
                                                learning_rate) # / (floor(itr/1000) + 1))
        #decent = abs(decent) * np.sign(grad)
        if itr%stop_itr == 0:
            if (abs((dTotalCost[itr-stop_itr-1:itr-1].sum(axis=0) - dTotalCost[itr]*stop_itr)) < threshold).all():
                if (abs(grad) < threshold).all():
                    msg = 'found solution'
                    break
                elif ((v[itr] == 1) * (dTotalCost[itr] < 0)).any() or ((v[itr] == epsilon) * (dTotalCost[itr] > 0)).any():
                    msg = 'no close solution'
                    break

        v[itr + 1] = v[itr] - decent  # np.minimum(np.maximum(dTotalCost * learning_rate, -0.01), 0.01)
        v[itr + 1] = np.minimum(np.maximum(v[itr + 1], epsilon), 1)

        pbar.set_postfix({"dv: ": dTotalCost[itr].sum(),
                          "Total_cost": TotalCost[itr].sum()})

    if solution_test and msg=='found solution':
        TotalCost_test = (outer['l'].reshape(groups, 1) * I_test + 1 / v_test - 1).sum(axis=1)

        sol = (TotalCost[itr].sum() - TotalCost_test.sum()) if gov else (TotalCost[itr][main_player_test] - TotalCost_test[main_player_test])

        test_results['solution'] = (sol < 0)

    return {'v': v[itr], 'v_der': dTotalCost[itr], 'cost': TotalCost[itr], 'msg': msg, 'test_results': test_results}
