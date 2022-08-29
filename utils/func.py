import numpy as np
from tqdm import tqdm
from math import floor


def infected(I, v, outer):
    new_I = (outer['beta'] * outer['d'] * v * v.T * (1 - I).reshape((outer['d'].shape[0], 1)) *
             I.reshape((1, outer['d'].shape[0]))).sum(axis=1)

    return I + new_I


def calculate_derivative_old(I, outer, v, groups, gov=False):
    if gov:
        diag = np.ones(groups) * 2
    else:
        diag = np.diag(np.ones(groups)) + np.ones(groups)
    x_multi = ((1 - I.T) @ I)
    dI = outer['beta'] * outer['d'] * x_multi * v.T * diag
    return dI


def derivative_test(dI, I, outer, v, groups, gov=False):
    dI_p = dI.copy()
    dI = dI.copy()
    d = outer['d']
    dI[0, 0, 0] = dI_p[0, 0, 0] + outer['beta'] * (d[0,1]*v[0,1]*v[1,0]*(dI_p[1,0,0]*(1-I[0]) - I[1]*dI_p[0,0,0]) +
                                                   d[0,0]*v[0,0]*v[0,0]*(dI_p[0,0,0]*(1-I[0]) - I[0]*dI_p[0,0,0]) +
                                                   d[0,0]*v[0,0]*(I[0]*(1-I[0]))*2)

    dI[0, 0, 1] = dI_p[0, 0, 1] + outer['beta'] * (d[0,1]*v[0,1]*v[1,0]*(dI_p[1,0,1]*(1-I[0]) - I[1]*dI_p[0,0,1]) +
                                                   d[0,0]*v[0,0]*v[0,0]*(dI_p[0,0,1]*(1-I[0]) - I[0]*dI_p[0,0,1]) +
                                                   d[0,1]*v[1,0]*(I[1]*(1-I[0])))

    dI[0, 1, 0] = dI_p[0, 1, 0] + outer['beta'] * (d[0,1]*v[0,1]*v[1,0]*(dI_p[1,1,0]*(1-I[0]) - I[1]*dI_p[0,1,0]) +
                                                   d[0,0]*v[0,0]*v[0,0]*(dI_p[0,1,0]*(1-I[0]) - I[0]*dI_p[0,1,0]) +
                                                   d[0,1]*v[0,1]*(I[1]*(1-I[0])))

    dI[0, 1, 1] = dI_p[0, 1, 1] + outer['beta'] * (d[0,1]*v[0,1]*v[1,0]*(dI_p[1,1,1]*(1-I[0]) - I[1]*dI_p[0,1,1]) +
                                                   d[0,0]*v[0,0]*v[0,0]*(dI_p[0,1,1]*(1-I[0]) - I[0]*dI_p[0,1,1])
                                                   )
    dI[1, 0, 0] = dI_p[1, 0, 0] + outer['beta'] * (d[1,0]*v[0,1]*v[1,0]*(dI_p[0,0,0]*(1-I[1]) - I[0]*dI_p[1,0,0]) +
                                                   d[1,1]*v[1,1]*v[1,1]*(dI_p[1,0,0]*(1-I[1]) - I[1]*dI_p[1,0,0])
                                                   )

    dI[1, 0, 1] = dI_p[1, 0, 1] + outer['beta'] * (d[1,0]*v[0,1]*v[1,0]*(dI_p[0,0,1]*(1-I[1]) - I[0]*dI_p[1,0,1]) +
                                                   d[1,1]*v[1,1]*v[1,1]*(dI_p[1,0,1]*(1-I[1]) - I[1]*dI_p[1,0,1]) +
                                                   d[1,0]*v[1,0]*(I[0]*(1-I[1])))

    dI[1, 1, 0] = dI_p[1, 1, 0] + outer['beta'] * (d[1,0]*v[0,1]*v[1,0]*(dI_p[0,1,0]*(1-I[1]) - I[0]*dI_p[1,1,0]) +
                                                   d[1,1]*v[1,1]*v[1,1]*(dI_p[1,1,0]*(1-I[1]) - I[1]*dI_p[1,1,0]) +
                                                   d[1,0]*v[0,1]*(I[0]*(1-I[1])))

    dI[1, 1, 1] = dI_p[1, 1, 1] + outer['beta'] * (d[1,0]*v[0,1]*v[1,0]*(dI_p[0,1,1]*(1-I[1]) - I[0]*dI_p[1,1,1]) +
                                                   d[1,1]*v[1,1]*v[1,1]*(dI_p[1,1,1]*(1-I[1]) - I[1]*dI_p[1,1,1]) +
                                                   d[1,1]*v[1,1]*(I[1]*(1-I[1]))*2)

    return dI


def calculate_derivative(dI, I, outer, v, groups, gov=False):

    def return_indx_and_diag(p, i, j, gov):
        if i == p and j == p:
            diag = 2
            indx = p
        elif i != p and j != p:
            diag = 0
            indx = 0
        else:
            diag = 1
            indx = i if i != p else j
        return diag, indx

    if gov:
        diag = np.ones(groups) * 2
    else:
        diag = np.diag(np.ones(groups)) + np.ones(groups)

    new_dI = dI.copy()
    if gov:
        for p in range(groups):
            new_dI[p] += outer['beta'] * (outer['d'][p, w] * v[p, w] * v[w, p] * (dI[w, i, j] * (1 - I[p]) -
                                                                                    I[w] * (dI[p, i, j])))
    else:
        for p in range(groups):
            for i in range(groups):
                for j in range(groups):
                    for w in range(groups):
                        new_dI[p, i, j] += outer['beta']*(outer['d'][p, w]*v[p, w]*v[w, p]*(dI[w, i, j] * (1-I[p]) -
                                                                                            I[w] * (dI[p, i, j])))
                    diag, indx = return_indx_and_diag(p, i, j, gov)

                    new_dI[p, i, j] += outer['beta']*outer['d'][p,indx]*v[j, i]*I[indx]*(1-I[p])*diag

    '''
    A = outer['beta'] * outer['d'] * v * v.T
    B = dI
    B = dI * (1 - I)
    I_multi = ((1 - I.T) @ I)
    dI_I_multi = ((1 - I.T) @ dI)
    I_dI_multi = ((1 - dI.T) @ I)
    v_multi = v * v.T
    dI = dI + outer['beta'] * outer['d'] * v_multi * (dI_I_multi - I_dI_multi + I_multi) * diag
    '''
    return new_dI


def adam_optimizer_iteration(grad, m, u, beta_1, beta_2, itr, epsilon, learning_rate):
    m = beta_1 * m + (1 - beta_1) * grad
    u = beta_2 * u + (1 - beta_2) * grad ** 2
    m_hat = m / (1 - beta_1 ** (itr + 1))
    u_hat = u / (1 - beta_2 ** (itr + 1))
    adam = (learning_rate * m_hat) / (u_hat ** 0.5 + epsilon)

    return adam, m, u


def optimize(T, I0, outer, gov=False, learning_rate=.01, max_itr=10000, epsilon=10**-8, beta_1=.9, beta_2=.999
             , Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=10**-6, only_finals=False, seed=None
             , leave=True):
    m = 0
    u = 0
    rand_gen = np.random.default_rng(seed)
    groups = outer['d'].shape[0]
    v = np.zeros((max_itr + 1, groups, groups))
    TotalCost = np.zeros((max_itr, groups))
    dTotalCost = np.zeros((max_itr, groups, groups))
    v[0] = rand_gen.random(1) if gov else rand_gen.random((groups, groups))
    dI_agg = np.zeros((groups, groups))

    msg = 'time out'
    pbar = tqdm(range(max_itr), leave=leave)
    for itr in pbar:
        dI = np.zeros(groups) if gov else np.zeros((groups, groups, groups))
        dI_test = dI.copy()
        I = np.zeros((T, groups))
        if Recovered_rate > 0:
            R = np.zeros((T, groups))
            R[0, :] = 0
        I[0, :] = I0

        for t in range(T-1):
            I[t + 1, :] = infected(I[t, :], v[itr], outer)
            if Recovered_rate > 0:
                R[t + 1, :] = I[t, :] * Recovered_rate
                I[t + 1, :] -= R[t + 1, :]
            if ReSusceptible_rate > 0:
                I[t + 1, :] -= I[t, :] * ReSusceptible_rate

            dI = calculate_derivative(dI, I[t], outer, v[itr], groups, gov=gov)

            dI_test = derivative_test(dI_test, I[t], outer, v[itr], groups, gov=gov)

        # InfectoionCost = outer['l'].reshape(groups, 1) * I
        # dI = calculate_derivative(I, outer, v[itr], groups, gov=gov)
        for i in range(groups):
            dI_agg[i, :] = dI[i, i, :]
        # Cost = 1/v -1
        '''
        i = 0
        j = 0
        I_test = np.zeros((T, groups))
        v_test = v[itr].copy()
        v_test[i, j] += epsilon/100000

        I_test[0, :] = I0
        I_test2 = I_test.copy()
        for t in range(T-1):
            I_test[t + 1, :] = infected(I_test[t, :], v_test, outer)

        dv_test = (I_test[T-1][i] - I[T-1][i])/(epsilon/100000)
        print(dI[i,i,j], dv_test, dI[i,i,j]-dv_test)
        '''
        dCost = -1 / v[itr] ** 2
        # [[ 9.471823    3.39994729], [52.37598457 23.10380243]]
        TotalCost[itr] = (outer['l'].reshape(groups, 1) * I[T - 1] + 1 / v[itr] - 1).sum(axis=1)
        # print(TotalCost)
        dTotalCost[itr] = outer['l'].reshape(groups, 1) * dI_agg + dCost
        grad = dTotalCost[itr].sum() if gov else dTotalCost[itr]
        decent, m, u = adam_optimizer_iteration(grad, m, u, beta_1, beta_2, itr, epsilon,
                                                learning_rate / (floor(itr/1000) + 1))
        #decent = abs(decent) * np.sign(grad)
        if itr%1000 == 0 and itr > 2000:
            pass
            #learning_rate /= 10
            # beta_1 -= 0.1
            # beta_2 -= 0.01
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
    if only_finals:
        return {'v': v[itr], 'v_der': dTotalCost[itr], 'cost': TotalCost[itr], 'msg': msg}
    else:
        return v[:itr], dTotalCost[:itr], TotalCost[:itr], msg
