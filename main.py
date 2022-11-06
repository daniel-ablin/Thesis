import numpy as np
from utils.func import infected, calculate_derivative, adam_optimizer_iteration, optimize
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import date


today = date.today()

learning_rate = 0.001

rng = 2
groups = 10
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
m, m_gov = 0, 0
u, u_gov = 0, 0
counter = 0
stop_itr = 50
Threshold = 10 ** -6
seed = 129
columns = ['T', 'I0', 'd', 'l', 'Recovered_rate', 'ReSusceptible_rate', 'sol', 'sol_gov', 'time']
data = list()
rnd = np.random.default_rng(seed)

rnd_search = True

if not rnd_search:
    T_list = rnd.integers(2, 1000, size=rng)
    I0_list = (0.1 - epsilon) * rnd.random(rng) + epsilon
    d_list = rnd.random((rng, groups, groups))/(groups**2)
    temp_list = rnd.integers(1, 10, size=rng)/groups
    l_list = rnd.integers(0, 10, size=rng)
    for T in T_list:
        for I0 in I0_list:
            for d in d_list:
                for temp, l_temp in zip(temp_list, l_list):
                    start = timer()
                    outer = {'beta': 2.3/30,
                             'd': d.copy(),
                             'l': np.array([temp*(1+l_temp*(i != 0)) for i in range(groups)])
                             }

                    Recovered_rate = 0
                    ReSusceptible_rate = 0

                    sol = optimize(T, I0, outer, one_v_for_all=True, learning_rate=.01, max_itr=1000, epsilon=10 ** -8,
                                   beta_1=.9
                                   , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50,
                                   threshold=Threshold
                                   , seed=seed, derv_test=True, solution_test=True, total_cost_test=True)

                    sol_gov = optimize(T, I0, outer, gov=True, learning_rate=.01, max_itr=1000, epsilon=10 ** -8,
                                       beta_1=.9
                                       , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50,
                                       threshold=Threshold
                                       , seed=seed, derv_test=True, solution_test=True, total_cost_test=True)

                    end = timer()
                    data.append(
                        [T, I0, outer['d'].copy(), outer['l'].copy(), Recovered_rate, ReSusceptible_rate, sol, sol_gov, end - start])
    data = pd.DataFrame(data, columns=columns)

    data.to_pickle(f'test_run_{today}_{seed}_{rng}.pickle')
    print('stop')

else:

    for itr in range(rng):
        print(f'\niteration: {itr}')
        start = timer()
        T = rnd.integers(2, 1000)
        I0 = (0.1 - epsilon) * rnd.random() + epsilon
        temp = rnd.integers(1, 10)/groups
        outer = {'beta': 2.3/30,
                 'd': rnd.random((groups, groups))/(groups**2),
                 'l': np.array([temp*(1+rnd.integers(0, 10)*(i != 0)) for i in range(groups)])
                 }
        Recovered_rate = 0
        ReSusceptible_rate = 0

        sol = optimize(T, I0, outer, one_v_for_all=True, learning_rate=.01, max_itr=1000, epsilon=10**-8, beta_1=.9
                       , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=Threshold
                       , seed=seed, derv_test=True, solution_test=True, total_cost_test=True)

        sol_gov = optimize(T, I0, outer, gov=True, learning_rate=.01, max_itr=1000, epsilon=10**-8, beta_1=.9
                           , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=Threshold
                           , seed=seed, derv_test=True, solution_test=True, total_cost_test=True)

        end = timer()
        data.append([T, I0, outer['d'], outer['l'], Recovered_rate, ReSusceptible_rate, sol, sol_gov, end - start])
    data = pd.DataFrame(data, columns=columns)

    data.to_pickle(f'test_run_{today}_{seed}_{itr}.pickle')
    print('stop')
