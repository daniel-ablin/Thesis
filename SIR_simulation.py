from datetime import datetime
from multiprocessing import Pool
from functools import partial
import pandas as pd

from utils.risk_initializer import RiskInitializer
from utils.utils import get_d_matrix
from utils.run_simulations import run_full_simulation

groups = 3
beta = 0.3/12.5
d, mean_age, norm_factor = get_d_matrix(groups, norm_to_one_meeting=True)
T = int(1.5 * 365 * norm_factor)
number_of_simulations = 1000
recovered_rate = 1/10 / norm_factor


if __name__ == '__main__':
    now = datetime.now()

    run_func = partial(run_full_simulation, groups=groups, T=T, beta=beta, recovered_rate=recovered_rate, d=d)

    risk_l = RiskInitializer(seed=41).age_to_risk_exponential_func(mean_age, number_of_simulations)

    #a = run_func(RiskInitializer(seed=41).age_to_risk_exponential_func(mean_age))

    with Pool(processes=10) as pool:
        data_list = pool.map(run_func, risk_l)

    data = pd.DataFrame(data_list)

    data.to_pickle(f'test_run_{number_of_simulations}_{groups}_{now}.pickle')
    print('stop')

