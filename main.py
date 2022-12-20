import numpy as np
from utils.func import optimize, get_d_matrix
import pandas as pd
from timeit import default_timer as timer
from datetime import date
from multiprocessing import Pool
import itertools


def run_model_random_search(itr):
    max = 0.3
    temp = (np.random.rand(groups)*(max-epsilon) + epsilon)
    temp = temp.cumsum()
    temp = np.multiply(temp, np.arange(1, groups*10, 10))*0.05
    outer = {'beta': 2.3 / 30,
             'd': d,
             'l': temp # np.cumsum(temp)
             }
    Recovered_rate = 1 / 14

    print(itr)
    return run_optimizers(T, I0, outer, Recovered_rate)


def run_model_linear_search(itr):
    d, l = itr
    outer = {'beta': 2.3 / 30,
             'd': d,
             'l': l
             }
    Recovered_rate = 1 / 14

    return run_optimizers(T, I0, outer, Recovered_rate)


def run_optimizers(T, I0, outer, Recovered_rate):
    start = timer()

    sol = optimize(T, I0, outer, one_v_for_all=True, learning_rate=learning_rate, max_itr=1000,
                   Recovered_rate=Recovered_rate, stop_itr=50, filter_elasticity=filter_elasticity)

    sol_gov = optimize(T, I0, outer, gov=True, learning_rate=learning_rate, max_itr=1000,
                       Recovered_rate=Recovered_rate, stop_itr=50, filter_elasticity=filter_elasticity)

    if groups > 2:
        sol_sec = optimize(T, I0, outer, one_v_for_all=True, learning_rate=learning_rate, max_itr=1000,
                           Recovered_rate=Recovered_rate, stop_itr=50,
                           filter_elasticity=filter_elasticity, sec_smallest_def=True)
    else:
        sol_sec = sol.copy()
    end = timer()
    return [T, I0, outer['d'], outer['l'], filter_elasticity, sol, sol_gov, sol_sec,
            end - start]


iter_counter = 0
learning_rate = 0.01
rng = 50
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
stop_itr = 35
Threshold = 10 ** -6
seed = 129
rnd = np.random.default_rng(seed)
groups = 2
d = get_d_matrix(groups)

I0 = 1/10000
T = int(1.5 * 365)
filter_elasticity = 1 #/ 8  # https://www.lonessmith.com/wp-content/uploads/2021/02/BSIR-nov.pdf page 7

if __name__ == '__main__':
    today = date.today()
    columns = ['T', 'I0', 'd', 'l', 'contagiousness', 'sol', 'sol_gov',
               'sol_sec', 'time']

    rnd_search = True
    run_model_random_search(2)

    with Pool() as pool:
        if rnd_search:
            data_list = pool.map(run_model_random_search, range(rng))
        else:
            d_list = rnd.random((rng, groups, groups)) / (groups ** 2)
            temp_list = rnd.integers(1, 10, size=rng) / groups
            l_list = rnd.integers(0, 10, size=rng)
            l = np.array([temp_list * (1 + l_list * (i != 0)) for i in range(groups)])
            params_list = itertools.product(d_list, l)
            data_list = pool.map(run_model_linear_search, params_list)

    data = pd.DataFrame(data_list, columns=columns)

    data.to_pickle(f'test_run_{today}_{seed}_{rng}.pickle')
    print('stop')
