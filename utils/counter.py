import numpy as np
from numpy.typing import NDArray
from utils.models_type_funcs import ModelFuncs


class LearningRateCounter:
    def __init__(self, counter_size: int):
        self.counter_size = counter_size
        self.counter_count = 0

    def update_learning_rate(self, dTotalCost: NDArray[float], itr: int, learning_rate: NDArray[float], model_funcs: ModelFuncs) -> NDArray[float]:
        if self.counter_count > self.counter_size:
            cond_nums = model_funcs.calc_condition_for_learning_rate_adjust(dTotalCost[itr - self.counter_size:itr])
            cond1 = np.where((cond_nums > 0.5).any(axis=0), 2, 1)
            cond2 = np.where((cond_nums < 0.05).all(axis=0), -0.1, 0)
            learning_rate /= cond1 + cond2
            self.counter_count = 0
        else:
            self.counter_count += 1

        return learning_rate

    def restart_counter(self):
        self.counter_count = 0
