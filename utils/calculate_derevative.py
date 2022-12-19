import numpy as np
from numba import jit
from itertools import product


#@jit(nopython=True)
def calculate_derivative(dS, dI, I, S, Recovered_rate, beta, d, v, groups, gov=False, one_v_for_all=False):
    if one_v_for_all:
        dS_new, dI_new = calculate_derivative_for_v_for_all(dS, dI, I, S, Recovered_rate, beta, d, v, groups)
        return np.diag(dS_new)[:, np.newaxis],  dS_new, dI_new

    elif gov:
        dS_new, dI_new = calculate_derivative_for_gov(dS, dI, I, S, Recovered_rate, beta, d, v, groups)
        return dS_new[:, np.newaxis], dS_new, dI_new

    else:
        dS_new, dI_new = calculate_derivative_for_v_for_each(dS, dI, I, S, Recovered_rate, beta, d, v, groups)
        dS_agg = np.zeros((groups, groups))
        for i in range(groups):
            dS_agg[i, :] = dS_new[i, i, :]
        return dS_agg, dS_new, dI_new


#@jit(nopython=True)
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


#@jit(nopython=True)
def calculate_derivative_for_v_for_each(dS, dI, I, S, Recovered_rate, beta, d, v, groups):

    new_dS = dS.copy()
    new_dI = dI.copy()
    derivative_delta = np.zeros(dS.shape)

    for p in range(groups):
        for i in range(groups):
            for j in range(groups):
                for w in range(groups):
                    derivative_delta[p, i, j] += beta*(d[p, w]*v[p, w]*v[w, p] *
                                                       (dI[w, i, j]*S[p] + I[w]*dS[p, i, j]))
                diag = (i == p) + (j == p)
                indx = i if i != p else j

                derivative_delta[p, i, j] += beta*d[p,indx]*v[indx, p]*I[indx]*(S[p])*diag

    return new_dS - derivative_delta, new_dI + derivative_delta - new_dI*Recovered_rate


#@jit(nopython=True)
def calc_diag(p, i, w):
        if p == i and p == w:
            return 2, p
        elif p != i and i != w:
            return 0, 0
        elif p == i and p != w:
            return 1, w
        else:
            return 1, p


#@jit(nopython=True)
def calculate_derivative_for_v_for_all(dS, dI, I, S, Recovered_rate, beta, d, v, groups):

    new_dS = dS.copy()
    new_dI = dI.copy()
    derivative_delta = np.zeros(dS.shape)

    '''def calc_diag(p, i, w):
        if p == i and p == w:
            return 2, p
        elif p != i and i != w:
            return 0, 0
        elif p == i and p != w:
            return 1, w
        else:
            return 1, p'''

    for p in range(groups):
        for i in range(groups):
            for w in range(groups):
                '''
                if p == i and p == w:
                    diag, indx = 2, p
                elif p != i and i != w:
                    diag, indx = 0, 0
                elif p == i and p != w:
                    diag, indx = 1, w
                else:
                    diag, indx = 1, p'''
                diag, indx = calc_diag(p, i, w)
                derivative_delta[p, i] += beta*d[p,w]*(v[p]*v[w]*(dI[w,i]*S[p] + I[w]*dS[p,i]) +
                                                       v[indx]*I[w]*S[p]*diag)

    return new_dS - derivative_delta, new_dI + derivative_delta - new_dI*Recovered_rate


#@jit(nopython=True)
def calculate_derivative_for_gov(dS, dI, I, S, Recovered_rate, beta, d, v, groups):
    new_dS = dS.copy()
    new_dI = dI.copy()
    derivative_delta = np.zeros(dS.shape)

    for p in range(groups):
        for i in range(groups):
            derivative_delta[p] += beta * d[p, i] * (v**2 * (dI[i]*(S[p]) + I[i]*dS[p]) + 2*v*(I[i] * S[p]))
    return new_dS - derivative_delta, new_dI + derivative_delta - new_dI*Recovered_rate


'''outer['beta']*outer['d'][p,w]*(v[p]*v[w]*(dI[w,i]*S[p] + I[w]*dS[p,i]) + v[indx]*I[w]*S[p]*diag)




a = outer['d'] * v[itr] * v[itr].T
b = a * S
c = a * I.reshape(T, 1, groups)
d = S * I.reshape(T, 1, groups)
e = np.broadcast_to(d, [groups] + list(d.shape))
shape = e.shape
N = shape[0]
e = e.flatten()
for n in range(N):
    e[n + n*groups + n*groups**2] *= 2*v[n]
        
    zero_index = [i + j*groups for i, j in product(range(groups), range(groups)) if i != n and j != n]
    e[zero_index] = 0
    
    zero_index = [i + j*groups for i, j in product(range(groups), range(groups)) if i != n and j != n]
    
    
'''