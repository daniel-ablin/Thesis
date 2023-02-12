import numpy as np
from utils.func import optimize, get_d_matrix
import pandas as pd
from timeit import default_timer as timer
from datetime import date
from multiprocessing import Pool
import itertools


def age_to_risk_exponential_func(base, ratio, age, exponent=6):
    return base*((ratio-1)*(age/100)**exponent + 1)


def run_model_random_search(itr):
    max = 1
    temp = (np.random.rand(groups)*(max-epsilon) + epsilon)
    temp = temp.cumsum()
    norm = groups + 1#100
    temp = np.multiply(temp, np.arange(1, groups*norm + 1, norm))*50

    a = np.random.uniform(0, 1000)
    b = np.random.uniform(0, 3000)
    temp = age_to_risk_exponential_func(a, b, mean_age)
    outer = {'beta': 0.3/12.5,
             'd': d,
             'l': temp # np.cumsum(temp)
             }
    Recovered_rate = 1 / 10

    print(itr)
    return run_optimizers(T, I0, outer, Recovered_rate)


def run_optimizers(T, I0, outer, Recovered_rate):
    start = timer()

    sol = optimize(T, I0, outer, one_v_for_all=True, learning_rate=learning_rate, max_itr=1000,
                   Recovered_rate=Recovered_rate, stop_itr=50, filter_elasticity=filter_elasticity)

    sol_gov = optimize(T, I0, outer, gov=True, learning_rate=learning_rate, max_itr=1000,
                       Recovered_rate=Recovered_rate, stop_itr=50, filter_elasticity=filter_elasticity)

    if False and groups > 2:
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
rng = 10000
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
stop_itr = 35
Threshold = 10 ** -6
seed = 129
rnd = np.random.default_rng(seed)
groups = 4
d, mean_age = get_d_matrix(groups)

I0 = 1/100000
T = int(1.5 * 365)
filter_elasticity = 1/8  # https://www.lonessmith.com/wp-content/uploads/2021/02/BSIR-nov.pdf page 7

if __name__ == '__main__':
    today = date.today()
    columns = ['T', 'I0', 'd', 'l', 'contagiousness', 'sol', 'sol_gov',
               'sol_sec', 'time']

    #run_model_random_search(2)

    with Pool(processes=10) as pool:
        data_list = pool.map(run_model_random_search, range(rng))

    data = pd.DataFrame(data_list, columns=columns)

    data.to_pickle(f'test_run_{today}_{seed}_{rng}_{groups}.pickle')
    print('stop')
