from datetime import datetime
from multiprocessing import Pool
from functools import partial
import pandas as pd

from utils.risk_initializer import RiskInitializer
from utils.utils import get_d_matrix, refactor_d_for_SI_simulation
from utils.run_simulations import run_full_SI_simulation

groups = 2
T = int(1.5 * 365)
beta = 0.3/12.5
d, mean_age = get_d_matrix(groups)
d, d_update_rule = refactor_d_for_SI_simulation(d)
number_of_simulations = 10000
recovered_rate = 0
max_d_itr = 25


if __name__ == '__main__':
    now = datetime.now()

    run_func = partial(run_full_SI_simulation, groups=groups, T=T, beta=beta, recovered_rate=recovered_rate, d=d,
                       d_update_rule=d_update_rule, max_itr=max_d_itr)

    risk_l = RiskInitializer().age_to_risk_exponential_func(mean_age, number_of_simulations)

    with Pool(processes=10) as pool:
        data_list = pool.map(run_func, risk_l)

    data = pd.DataFrame(data_list)

    data.to_pickle(f'test_run_{number_of_simulations}_{groups}_{now}.pickle')
    print('stop')
