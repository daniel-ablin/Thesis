import numpy as np

def infected(I, v, outer):
    new_I = (outer['beta'] * outer['d'] * v * v.T * (1 - I).reshape((outer['d'].shape[0], 1)) *
             I.reshape((1, outer['d'].shape[0]))).sum(axis=1)

    return I + new_I

def calc_dinamics(v, T, I0, groups, Recovered_rate, gov, derv_test, solution_test, ReSusceptible_rate):
    dI = np.zeros(groups) if gov else np.zeros((groups, groups)) if one_v_for_all else np.zeros(
        (groups, groups, groups))
    I = np.zeros((T, groups))
    I[0, :] = I0

    if derv_test or solution_test:
        test_results = dict()
        I_test = I[0, :].copy()
        v_test = v[itr].copy()
        main_player_test, sub_player_test = np.random.choice(groups, 2)
        if gov:
            v_test += epsilon
        else:
            v_test[main_player_test, sub_player_test] += epsilon
    else:
        test_results = None

    if Recovered_rate > 0:
        R = np.zeros((T, groups))
        R[0, :] = 0

    for t in range(T - 1):
        I[t + 1, :] = infected(I[t, :], v[itr], outer)
        if derv_test or solution_test:
            I_test = infected(I_test, v_test, outer)
        if Recovered_rate > 0:
            R[t + 1, :] = I[t, :] * Recovered_rate
            I[t + 1, :] -= R[t + 1, :]
        if ReSusceptible_rate > 0:
            I[t + 1, :] -= I[t, :] * ReSusceptible_rate