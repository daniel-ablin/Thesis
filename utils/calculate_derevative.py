import numpy as np
from numba import jit
from itertools import product


#@jit(nopython=True)
def calculate_derivative(dS, dI, I, S, Recovered_rate, beta, d, v, groups, gov=False, one_v_for_all=False):
    if one_v_for_all:
        dS_new, dI_new = calculate_derivative_for_v_for_all(dS, dI, I, S, Recovered_rate, beta, d, v, groups)
        return np.expand_dims(np.diag(dS_new), -1),  dS_new, dI_new

    elif gov:
        dS_new, dI_new = calculate_derivative_for_gov(dS, dI, I, S, Recovered_rate, beta, d, v, groups)
        return np.expand_dims(dS_new,  -1), dS_new, dI_new

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
                                                       (dI[w, i, j]*S[p] + I[w]*dS[p, i, j]))[0]
                diag = (i == p) + (j == p)
                indx = i if i != p else j

                derivative_delta[p, i, j] += beta*d[p,indx]*v[indx, p]*I[indx]*(S[p])*diag[0]

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

    for p in range(groups):
        for i in range(groups):
            for w in range(groups):
                diag, indx = calc_diag(p, i, w)
                derivative_delta[p, i] += beta*d[p,w]*(v[p]*v[w]*(dI[w,i]*S[p] + I[w]*dS[p,i]) +
                                                       v[indx]*I[w]*S[p]*diag)[0]

    return new_dS - derivative_delta, new_dI + derivative_delta - new_dI*Recovered_rate


#@jit(nopython=True)
def calculate_derivative_for_gov(dS, dI, I, S, Recovered_rate, beta, d, v, groups):
    new_dS = dS.copy()
    new_dI = dI.copy()
    derivative_delta = np.zeros(dS.shape)

    for p in range(groups):
        for i in range(groups):
            derivative_delta[p] += beta * d[p, i] * (v**2 * (dI[i]*(S[p]) + I[i]*dS[p]) + 2*v*(I[i] * S[p]))[0]
    return new_dS - derivative_delta, new_dI + derivative_delta - new_dI*Recovered_rate


'''




a =  beta * d * v[itr] * v[itr].T
b = a * S
c = a * I.reshape(T, 1, groups)
f = S * I.reshape(T, 1, groups)
e = np.broadcast_to(f[:, np.newaxis], list(f.shape) + [groups]).T.copy()

indices = np.indices((groups, groups, groups))
unequal_index = (indices[2]!=indices[1]) & (indices[2]!=indices[0])
e[unequal_index] = 0

all_equal_index = (indices[0]==indices[2]) & (indices[1]==indices[2])
e[all_equal_index] = e[all_equal_index]*2*v[itr]

same_player_derivative_with_other_group_interaction_index = (indices[2]==indices[1]) & (indices[2]!=indices[0])
e[same_player_derivative_with_other_group_interaction_index] = e[same_player_derivative_with_other_group_interaction_index
]*(same_player_derivative_with_other_group_interaction_index*v[itr].T
  )[same_player_derivative_with_other_group_interaction_index, np.newaxis]

else_index = ~(unequal_index | all_equal_index | same_player_derivative_with_other_group_interaction_index)
e[else_index] = e[else_index]*(else_index*v[itr])[else_index, np.newaxis]

e = e.sum(axis=0).T

for i in range(e.shape[0]):
    



outer['beta']*outer['d'][p,w]*(v[p]*v[w]*(dI[w,i]*S[p] + I[w]*dS[p,i]) + v[indx]*I[w]*S[p]*diag)
shape = e.shape
N = shape[0]
e = e.flatten()
for n in range(N):
    e[n + n*groups + n*groups**2] *= 2*v[n]
        
    zero_index = [i + j*groups for i, j in product(range(groups), range(groups)) if i != n and j != n]
    e[zero_index] = 0
    
    zero_index = [i + j*groups for i, j in product(range(groups), range(groups)) if i != n and j != n]
    
    
'''