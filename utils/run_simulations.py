from timeit import default_timer as timer

from utils.model_optimizer import ModelOptimizer
from utils.model_types import ModelsTypes
from utils.utils import get_populations_proportions


def run_full_simulation(risk_l, groups, T, beta, recovered_rate, d, populations_proportions=None):
    start = timer()
    optimizers_args = dict(recovered_rate=recovered_rate,
                           populations_proportions=populations_proportions,
                           I0=1/100000,
                           max_itr=1000,
                           filter_elasticity=1/8
                           )
    simulator_args = dict(learning_rate=.01,
                          epsilon=10 ** -8,
                          stop_itr=50,
                          threshold=10 ** -6
                          )

    optimizer = ModelOptimizer(groups, T, beta, d, risk_l, ModelsTypes.anarchy, **optimizers_args)
    gov_optimizer = ModelOptimizer(groups, T, beta, d, risk_l, ModelsTypes.gov, **optimizers_args)

    test_results, msg, sol = optimizer.optimize(**simulator_args)
    gov_test_results, gov_msg, gov_sol = gov_optimizer.optimize(**simulator_args)

    end = timer()

    return dict(T=T, risk_l=risk_l, d=d, sol=sol, sol_gov=gov_sol, time=end - start)


def run_full_SI_simulation(risk_l, groups, T, beta, recovered_rate, d, d_update_rule, max_itr, populations_proportions):
    res_list = []
    d_local = d.copy()
    for i in range(max_itr):
        res_list.append(run_full_simulation(risk_l, groups, T, beta, recovered_rate, d_local.copy(), populations_proportions=populations_proportions))

        d_local += (100 / max_itr / 100) * d_update_rule

    return res_list
