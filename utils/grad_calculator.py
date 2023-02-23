import numpy as np
from model_types import ModelsTypes
from utils.data_classes import CostVariables


class GradCalculator:
    def get_grad(self):
        pass


class GovGradCalculator(GradCalculator):
    @staticmethod
    def calculate_grad(cost: CostVariables, itr, learning_rate, populations_proportions):
        grad = (np.nan_to_num(cost.dTotalCost[itr], posinf=1 / learning_rate,
                                  neginf=-1 / learning_rate) * populations_proportions).sum()

        return grad


class AnarchyGradCalculator(GradCalculator):
    @staticmethod
    def calculate_grad(cost: CostVariables, itr, learning_rate, populations_proportions):
        grad = cost.dTotalCost[itr]

        return grad


def calculate_grad(model_type: ModelsTypes, cost: CostVariables, itr, learning_rate, populations_proportions):
    factors = {ModelsTypes.anarchy: AnarchyGradCalculator,
               ModelsTypes.gov: GovGradCalculator}
    return factors[model_type].calculate_grad(cost, itr, learning_rate, populations_proportions)
