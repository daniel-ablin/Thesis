import numpy as np
from utils.func import infected, calculate_derivative, adam_optimizer_iteration, optimize
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import date
from multiprocessing import Pool
import itertools


def run_model_random_search(itr):
    rnd = np.random.default_rng(itr+2)
    # T = rnd.integers(2, 1000)
    # I0 = (0.1 - epsilon) * rnd.random() + epsilon
    # contagiousness = max(rnd.random(), epsilon)
    temp = rnd.integers(1, 100) / groups
    outer = {'beta': 2.3 / 30,
             'd': d,
             'l': np.array([temp * (1 + rnd.integers(0, 10) * (i != 0)) for i in range(groups)])
             }
    Recovered_rate = 0 # 1 / 17
    ReSusceptible_rate = 0

    print(itr)
    return run_optimizers(T, I0, outer, ReSusceptible_rate, Recovered_rate)


def run_model_linear_search(itr):
    d, l = itr
    outer = {'beta': 2.3 / 30,
             'd': d,
             'l': l
             }
    Recovered_rate = 1 / 14
    ReSusceptible_rate = 1 / 60

    return run_optimizers(T, I0, outer, ReSusceptible_rate, Recovered_rate)


def run_optimizers(T, I0, outer, ReSusceptible_rate, Recovered_rate):
    start = timer()

    sol = optimize(T, I0, outer, one_v_for_all=True, learning_rate=.01, max_itr=1000, epsilon=10 ** -8, beta_1=.9,
                   beta_2=.999, Recovered_rate=Recovered_rate, ReSusceptible_rate=ReSusceptible_rate, stop_itr=50,
                   threshold=Threshold, seed=seed, derv_test=True, solution_test=True, total_cost_test=True,
                   filter_elasticity=filter_elasticity)

    sol_gov = optimize(T, I0, outer, gov=True, learning_rate=.01, max_itr=1000, epsilon=10 ** -8, beta_1=.9,
                       beta_2=.999, Recovered_rate=Recovered_rate, ReSusceptible_rate=ReSusceptible_rate, stop_itr=50,
                       threshold=Threshold, seed=seed, derv_test=True, solution_test=True, total_cost_test=True,
                       filter_elasticity=filter_elasticity)
    end = timer()
    return [T, I0, outer['d'], outer['l'], Recovered_rate, ReSusceptible_rate, filter_elasticity, sol, sol_gov, end - start]


iter_counter = 0
learning_rate = 0.001
rng = 300
groups = 2
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
stop_itr = 50
Threshold = 10 ** -6
seed = 129
rnd = np.random.default_rng(seed)
d = np.array([[13.96027149, 2.876149293], [3.055542336, 2.004033465]])

I0 = 1/10000
T = int(1.5 * 365)
filter_elasticity = 1 / 8  # https://www.lonessmith.com/wp-content/uploads/2021/02/BSIR-nov.pdf page 7

if __name__ == '__main__':
    today = date.today()
    columns = ['T', 'I0', 'd', 'l', 'Recovered_rate', 'ReSusceptible_rate', 'contagiousness', 'sol', 'sol_gov', 'time']

    rnd_search = True
    run_model_random_search(2)

    with Pool() as pool:
        if rnd_search:
            data_list = pool.map(run_model_random_search, range(rng))
        else:
            #T_list = rnd.integers(2, 1000, size=rng)
            #I0_list = (0.1 - epsilon) * rnd.random(rng) + epsilon
            d_list = rnd.random((rng, groups, groups)) / (groups ** 2)
            temp_list = rnd.integers(1, 10, size=rng) / groups
            l_list = rnd.integers(0, 10, size=rng)
            l = np.array([temp_list * (1 + l_list * (i != 0)) for i in range(groups)])
            params_list = itertools.product(d_list, l)
            data_list = pool.map(run_model_linear_search, params_list)

    data = pd.DataFrame(data_list, columns=columns)

    data.to_pickle(f'test_run_{today}_{seed}_{rng}.pickle')
    print('stop')
