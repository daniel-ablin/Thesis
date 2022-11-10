import numpy as np

def calculate_derivative(dS, dI, I, S, Recovered_rate, outer, v, groups, gov=False, one_v_for_all=False):
    if one_v_for_all:
        dS_new, dI_new = calculate_derivative_for_v_for_all(dS, dI, I, S, Recovered_rate, outer, v, groups)
        return np.diag(dS_new)[:, np.newaxis],  dS_new, dI_new

    elif gov:
        dS_new, dI_new = calculate_derivative_for_gov(dS, dI, I, S, Recovered_rate, outer, v, groups)
        return dS_new[:, np.newaxis], dS_new, dI_new

    else:
        dS_new, dI_new = calculate_derivative_for_v_for_each(dS, dI, I, S, Recovered_rate, outer, v, groups)
        dS_agg = np.zeros((groups, groups))
        for i in range(groups):
            dS_agg[i, :] = dS_new[i, i, :]
        return dS_agg, dS_new, dI_new


def return_indx_and_diag(p, i, j):
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


def calculate_derivative_for_v_for_each(dS, dI, I, S, Recovered_rate, outer, v, groups):

    new_dS = dS.copy()
    new_dI = dI.copy()
    derivative_delta = np.zeros(dS.shape)

    for p in range(groups):
        for i in range(groups):
            for j in range(groups):
                for w in range(groups):
                    derivative_delta[p, i, j] += outer['beta']*(outer['d'][p, w]*v[p, w]*v[w, p] *
                                                                (dI[w, i, j]*S[p] + I[w]*dS[p, i, j]))
                diag = (i == p) + (j == p)
                indx = i if i != p else j

                derivative_delta[p, i, j] += outer['beta']*outer['d'][p,indx]*v[indx, p]*I[indx]*(S[p])*diag

    return new_dS - derivative_delta, new_dI + derivative_delta - new_dI*Recovered_rate


def calculate_derivative_for_v_for_all(dS, dI, I, S, Recovered_rate, outer, v, groups):

    new_dS = dS.copy()
    new_dI = dI.copy()
    derivative_delta = np.zeros(dS.shape)

    def calc_diag(p, i, w):
        if p == i and p == w:
            return 2, p
        elif p != i and p == w:
            return 0, 0
        elif p == i and p != w:
            return 1, w
        else:
            return 1, p

    for p in range(groups):
        for i in range(groups):
            for w in range(groups):
                diag, indx = calc_diag(p, i, w)
                derivative_delta[p, i] += outer['beta']*outer['d'][p,w]*(v[p]*v[w]*(dI[w,i]*S[p] + I[w]*dS[p,i]) +
                                                               v[indx]*I[w]*S[p]*diag)

    return new_dS - derivative_delta, new_dI + derivative_delta - new_dI*Recovered_rate


def calculate_derivative_for_gov(dS, dI, I, S, Recovered_rate, outer, v, groups):
    new_dS = dS.copy()
    new_dI = dI.copy()
    derivative_delta = np.zeros(dS.shape)

    for p in range(groups):
        for i in range(groups):
            derivative_delta[p] += outer['beta'] * outer['d'][p, i] * (v**2 * (dI[i]*(S[p]) + I[i]*dS[p])
                                                                       + 2*v*(I[i] * S[p]))
    return new_dS - derivative_delta, new_dI + derivative_delta - new_dI*Recovered_rate

def derivative_test(dI, I, S, outer, v, groups, gov=False):
    dI_p = dI.copy()
    dI = dI.copy()
    d = outer['d']
    dI[0, 0, 0] = dI_p[0, 0, 0] + outer['beta'] * (d[0,1]*v[0,1]*v[1,0]*(dI_p[1,0,0]*(S[0]) - I[1]*dI_p[0,0,0]) +
                                                   d[0,0]*v[0,0]*v[0,0]*(dI_p[0,0,0]*(S[0]) - I[0]*dI_p[0,0,0]) +
                                                   d[0,0]*v[0,0]*(I[0]*(S[0]))*2)

    dI[0, 0, 1] = dI_p[0, 0, 1] + outer['beta'] * (d[0,1]*v[0,1]*v[1,0]*(dI_p[1,0,1]*(S[0]) - I[1]*dI_p[0,0,1]) +
                                                   d[0,0]*v[0,0]*v[0,0]*(dI_p[0,0,1]*(S[0]) - I[0]*dI_p[0,0,1]) +
                                                   d[0,1]*v[1,0]*(I[1]*(S[0])))

    dI[0, 1, 0] = dI_p[0, 1, 0] + outer['beta'] * (d[0,1]*v[0,1]*v[1,0]*(dI_p[1,1,0]*(S[0]) - I[1]*dI_p[0,1,0]) +
                                                   d[0,0]*v[0,0]*v[0,0]*(dI_p[0,1,0]*(S[0]) - I[0]*dI_p[0,1,0]) +
                                                   d[0,1]*v[0,1]*(I[1]*(S[0])))

    dI[0, 1, 1] = dI_p[0, 1, 1] + outer['beta'] * (d[0,1]*v[0,1]*v[1,0]*(dI_p[1,1,1]*(S[0]) - I[1]*dI_p[0,1,1]) +
                                                   d[0,0]*v[0,0]*v[0,0]*(dI_p[0,1,1]*(S[0]) - I[0]*dI_p[0,1,1])
                                                   )
    dI[1, 0, 0] = dI_p[1, 0, 0] + outer['beta'] * (d[1,0]*v[0,1]*v[1,0]*(dI_p[0,0,0]*(S[1]) - I[0]*dI_p[1,0,0]) +
                                                   d[1,1]*v[1,1]*v[1,1]*(dI_p[1,0,0]*(S[1]) - I[1]*dI_p[1,0,0])
                                                   )

    dI[1, 0, 1] = dI_p[1, 0, 1] + outer['beta'] * (d[1,0]*v[0,1]*v[1,0]*(dI_p[0,0,1]*(S[1]) - I[0]*dI_p[1,0,1]) +
                                                   d[1,1]*v[1,1]*v[1,1]*(dI_p[1,0,1]*(S[1]) - I[1]*dI_p[1,0,1]) +
                                                   d[1,0]*v[1,0]*(I[0]*(S[1])))

    dI[1, 1, 0] = dI_p[1, 1, 0] + outer['beta'] * (d[1,0]*v[0,1]*v[1,0]*(dI_p[0,1,0]*(S[1]) - I[0]*dI_p[1,1,0]) +
                                                   d[1,1]*v[1,1]*v[1,1]*(dI_p[1,1,0]*(S[1]) - I[1]*dI_p[1,1,0]) +
                                                   d[1,0]*v[0,1]*(I[0]*(S[1])))

    dI[1, 1, 1] = dI_p[1, 1, 1] + outer['beta'] * (d[1,0]*v[0,1]*v[1,0]*(dI_p[0,1,1]*(S[1]) - I[0]*dI_p[1,1,1]) +
                                                   d[1,1]*v[1,1]*v[1,1]*(dI_p[1,1,1]*(S[1]) - I[1]*dI_p[1,1,1]) +
                                                   d[1,1]*v[1,1]*(I[1]*(S[1]))*2)

    return dI

def calculate_derivative_old(I, outer, v, groups, gov=False):
    if gov:
        diag = np.ones(groups) * 2
    else:
        diag = np.diag(np.ones(groups)) + np.ones(groups)
    x_multi = ((1 - I.T) @ I)
    dI = outer['beta'] * outer['d'] * x_multi * v.T * diag
    return dI
