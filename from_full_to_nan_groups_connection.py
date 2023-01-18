import numpy as np
from utils.func import optimize, get_d_matrix
import pandas as pd
from timeit import default_timer as timer
from datetime import date
from multiprocessing import Pool
import itertools


def rum_model_d_iter(itr):
    max_itr = 25

    max = 0.3
    temp = (np.random.rand(groups)*(max-epsilon) + epsilon)
    temp = temp.cumsum()
    norm = groups
    temp = np.multiply(temp, np.arange(1, groups*norm + 1, norm))*4
    Recovered_rate = 0
    print(itr)
    res_list = []
    d_local = d.copy()
    for i in range(max_itr):

        outer = {'beta': 2.3 / 30,
                 'd': d_local.copy(),
                 'l': temp # np.cumsum(temp)
                 }

        res_list.append(run_optimizers(T, I0, outer, Recovered_rate))

        d_local += (100 / max_itr / 100) * d_update_rule

    return res_list



def run_optimizers(T, I0, outer, Recovered_rate):
    start = timer()

    sol = optimize(T, I0, outer, one_v_for_all=True, learning_rate=learning_rate, max_itr=10000,
                   Recovered_rate=Recovered_rate, stop_itr=50, filter_elasticity=filter_elasticity)

    sol_gov = optimize(T, I0, outer, gov=True, learning_rate=learning_rate, max_itr=10000,
                       Recovered_rate=Recovered_rate, stop_itr=50, filter_elasticity=filter_elasticity)

    end = timer()
    return [T, I0, outer['d'], outer['l'], filter_elasticity, sol, sol_gov, end - start]


iter_counter = 0
learning_rate = 0.01
rng = 20
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
stop_itr = 35
Threshold = 10 ** -6
seed = 129
rnd = np.random.default_rng(seed)
groups = 2
d = get_d_matrix(groups)
d_base = d.sum(axis=1)/2
d.fill(0)
np.fill_diagonal(d, d_base)
d_update_rule = np.ones(d.shape) * d_base
np.fill_diagonal(d_update_rule, -d_base)

I0 = 1/10000
T = int(1.5 * 365)
filter_elasticity = 1 #/ 8  # https://www.lonessmith.com/wp-content/uploads/2021/02/BSIR-nov.pdf page 7

if __name__ == '__main__':
    today = date.today()
    columns = ['T', 'I0', 'd', 'l', 'contagiousness', 'sol', 'sol_gov', 'time']

    rnd_search = True
    rum_model_d_iter(2)

    with Pool() as pool:
        if rnd_search:
            data_list = pool.map(rum_model_d_iter, range(rng))
        else:
            d_list = rnd.random((rng, groups, groups)) / (groups ** 2)
            temp_list = rnd.integers(1, 10, size=rng) / groups
            l_list = rnd.integers(0, 10, size=rng)
            l = np.array([temp_list * (1 + l_list * (i != 0)) for i in range(groups)])
            params_list = itertools.product(d_list, l)
            data_list = pool.map(rum_model_d_iter, params_list)

    data_list = sum(data_list, [])

    data = pd.DataFrame(data_list, columns=columns)

    data.to_pickle(f'test_run_d_{today}_{seed}_{rng}.pickle')
    print('stop')
