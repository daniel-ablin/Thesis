import numpy as np
from utils.func import infected, calculate_derivative, adam_optimizer_iteration, optimize
import pandas as pd
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import date


today = date.today()

learning_rate = 0.001

rng = 10000
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
m, m_gov = 0, 0
u, u_gov = 0, 0
counter = 0
stop_itr = 50
Threshold = 10 ** -6
seed = 20
columns = ['T', 'I0', 'd', 'l', 'Recovered_rate', 'ReSusceptible_rate', 'sol', 'sol_gov', 'time']
data = list()
rnd = np.random.default_rng(seed)
for itr in range(5000):
    print(f'\niteration: {itr}')
    start = timer()
    T = rnd.integers(1, 1000)
    I0 = (0.1 - epsilon) * rnd.random() + epsilon
    temp = rnd.integers(1, 100)
    outer = {'beta': 2.3/30,
             'd': rnd.random((2, 2)),
             'l': np.array([temp, temp * rnd.integers(1, 50)])
             }
    Recovered_rate = 0
    ReSusceptible_rate = 0

    groups = outer['d'].shape[0]

    sol = optimize(T, I0, outer, gov=False, learning_rate=.01, max_itr=10000, epsilon=10**-8, beta_1=.9
                   , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=Threshold
                   , only_finals=True, seed=seed, derv_test=True, solution_test=True)

    sol_gov = optimize(T, I0, outer, gov=True, learning_rate=.01, max_itr=10000, epsilon=10**-8, beta_1=.9
                       , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=Threshold
                       , only_finals=True, seed=seed, derv_test=True, solution_test=True)

    end = timer()
    data.append([T, I0, outer['d'], outer['l'], Recovered_rate, ReSusceptible_rate, sol, sol_gov, end - start])
data = pd.DataFrame(data, columns=columns)

data.to_pickle(f'test_run_{today}_{seed}_{itr}.pickle')
print('stop')