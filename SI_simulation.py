from datetime import datetime
from multiprocessing import Pool
from functools import partial
import pandas as pd

from utils.risk_initializer import RiskInitializer
from utils.utils import get_d_matrix, refactor_d_for_SI_simulation, get_populations_proportions
from utils.run_simulations import run_full_SI_simulation

groups = 2
beta = 0.3/12.5
d, mean_age, norm_factor = get_d_matrix(groups, norm_to_one_meeting=True)
populations_proportions = get_populations_proportions(d)
d, d_update_rule = refactor_d_for_SI_simulation(d, fact=1)

T = int(100 * norm_factor)

recovered_rate = 0
max_d_itr = 25
number_of_simulations = 14


if __name__ == '__main__':
    now = datetime.now()

    run_func = partial(run_full_SI_simulation, groups=groups, T=T, beta=beta, recovered_rate=recovered_rate, d=d,
                       d_update_rule=d_update_rule, max_itr=max_d_itr, populations_proportions=populations_proportions)

    #a = run_func(RiskInitializer().age_to_risk_exponential_func(mean_age, 1))

    risk_l = RiskInitializer().age_to_risk_exponential_func(mean_age, number_of_simulations)

    with Pool(processes=10) as pool:
        data_list = pool.map(run_func, risk_l)

    data_list = sum(data_list, [])

    data = pd.DataFrame(data_list)

    data.to_pickle(f'test_run_{max_d_itr}_{groups}_{now}.pickle')
    print('stop')
