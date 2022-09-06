import numpy as np

def calculate_derivative(dI, I, outer, v, groups, gov=False, one_v_for_all=False):
    if one_v_for_all:
        dI_new = calculate_derivative_for_v_for_all(dI, I, outer, v, groups)
        return np.diag(dI_new)[:, np.newaxis],  dI_new

    elif gov:
        dI_new = calculate_derivative_for_gov(dI, I, outer, v, groups)
        return dI_new[:, np.newaxis], dI_new

    else:
        dI_new = calculate_derivative_for_v_for_each(dI, I, outer, v, groups)
        dI_agg = np.zeros((groups, groups))
        for i in range(groups):
            dI_agg[i, :] = dI_new[i, i, :]
        return dI_agg, dI_new


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


def calculate_derivative_for_v_for_each(dI, I, outer, v, groups):

    new_dI = dI.copy()

    for p in range(groups):
        for i in range(groups):
            for j in range(groups):
                for w in range(groups):
                    new_dI[p, i, j] += outer['beta']*(outer['d'][p, w]*v[p, w]*v[w, p]*(dI[w, i, j] * (1-I[p]) -
                                                                                        I[w] * (dI[p, i, j])))
                diag = (i == p) + (j == p)
                indx = i if i != p else j

                new_dI[p, i, j] += outer['beta']*outer['d'][p,indx]*v[indx, p]*I[indx]*(1-I[p])*diag

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


def calculate_derivative_for_v_for_all(dI, I, outer, v, groups):

    new_dI = dI.copy()

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
                new_dI[p, i] += outer['beta']*outer['d'][p,w]*(v[p]*v[w]*(dI[w,i]*(1-I[p]) - I[w]*(dI[p,i])) +
                                                               v[indx]*I[w]*(1-I[p])*diag)
            #diag, indx = return_indx_and_diag(p, i, p)

            #new_dI[p, i] += outer['beta']*outer['d'][p,indx]*v[indx]*I[indx]*(1-I[p])*diag

    return new_dI


def calculate_derivative_for_gov(dI, I, outer, v, groups):
    new_dI = dI.copy()

    for p in range(groups):
        for i in range(groups):
            new_dI[p] += outer['beta'] * outer['d'][p, i] * (v**2 * (dI[i]*(1-I[p]) - I[i]*(dI[p])) + 2*v*(I[i] * (1-I[p])))
    return new_dI

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

def calculate_derivative_old(I, outer, v, groups, gov=False):
    if gov:
        diag = np.ones(groups) * 2
    else:
        diag = np.diag(np.ones(groups)) + np.ones(groups)
    x_multi = ((1 - I.T) @ I)
    dI = outer['beta'] * outer['d'] * x_multi * v.T * diag
    return dI
