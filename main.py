import numpy as np
from utils.func import infected, calculate_derivative, adam_optimizer_iteration, optimize
from tqdm import tqdm
from math import floor

Recovered_rate = 0
ReSusceptible_rate = 0

learning_rate = 0.01
T = 100
I0 = 0.01
outer = {'beta': 2.3/30,
         'd': np.array(([0.7, 0.1], [0.1, 0.7])),
         'l': np.array([5, 6]),
         }

groups = outer['d'].shape[0]

rng = 10000
epsilon = 10**-8
beta_1 = 0.9
beta_2 = 0.999
m, m_gov = 0, 0
u, u_gov = 0, 0
counter = 0
stop_itr = 50
Threshold = 10 ** -6
test = 10

v, dTotalCost, TotalCost = optimize(T, I0, outer, gov=False, learning_rate=.01, max_itr=10000, epsilon=10**-8, beta_1=.9
                                    , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=10**-6
                                    , seed=None)

v_gov, dTotalCost_gov, TotalCost_gov = optimize(T, I0, outer, gov=True, learning_rate=.01, max_itr=10000, epsilon=10**-8, beta_1=.9
                                    , beta_2=.999, Recovered_rate=0, ReSusceptible_rate=0, stop_itr=50, threshold=10**-6
                                    , seed=None)


print('stop')