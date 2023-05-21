from datetime import datetime
from multiprocessing import Pool
from functools import partial
import pandas as pd

from utils.risk_initializer import RiskInitializer
from utils.utils import get_d_matrix, norm_d_to_one, get_populations_proportions
from utils.run_simulations import run_full_simulation

groups = 2
base_d, mean_age, norm_factor = get_d_matrix(groups, norm_to_one_meeting=True)
populations_proportions = get_populations_proportions(base_d)
T = int(1.5 * 365 * norm_factor)
number_of_simulations = 1000
recovered_rate = 1/10 / norm_factor
now = datetime.now()


if __name__ == '__main__':
    for base_beta in [.4, .5, .6, .2, .1, .3]:
        beta = base_beta / 12.5
        d, beta = norm_d_to_one(base_d, beta)

        run_func = partial(run_full_simulation, groups=groups, T=T, beta=beta, recovered_rate=recovered_rate, d=d, populations_proportions=populations_proportions)

        risk_l = RiskInitializer(seed=41).age_to_risk_exponential_func(mean_age, number_of_simulations)

        # a = run_func(RiskInitializer(seed=41).age_to_risk_exponential_func(mean_age))

        with Pool(processes=5) as pool:
            data_list = pool.map(run_func, risk_l)

        data = pd.DataFrame(data_list)

        data.to_pickle(f'test_run_{number_of_simulations}_{groups}_{base_beta}_{now}.pickle')
        print('stop')

