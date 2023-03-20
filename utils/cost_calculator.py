from numpy.typing import NDArray

from utils.data_classes import CostVariables, ModelOuterVariables


class CostCalculator:
    def __init__(self, model_outer_variables: ModelOuterVariables, const=-2):
        self.elasticity_adjust = model_outer_variables.elasticity_adjust
        self.groups = model_outer_variables.groups
        self.const = const
        self.risk_l = model_outer_variables.risk_l

    def calc_d_cost(self, v: NDArray[float]) -> NDArray[float]:
        elasticity_adjust = self.elasticity_adjust
        return -elasticity_adjust / (v ** (elasticity_adjust + 1)) + elasticity_adjust

    def calc_total_cost(self, S: NDArray[float], v: NDArray[float]) -> NDArray[float]:
        elasticity_adjust = self.elasticity_adjust
        const = self.const
        return self.risk_l.reshape(self.groups,
                                   1) * S ** const + 1 / v ** elasticity_adjust + elasticity_adjust * v - elasticity_adjust - 1

    def calc_d_total_cost(self, dS_agg: NDArray[float], S: NDArray[float], v: NDArray[float]) -> NDArray[float]:
        const = self.const
        dCost = self.calc_d_cost(v)
        return self.risk_l.reshape(self.groups, 1) * const * dS_agg * S ** (const - 1) + dCost

    def calculate(self, S: NDArray[float], v: NDArray[float], dS_agg: NDArray[float]) -> CostVariables:
        TotalCost = self.calc_total_cost(S, v)
        dTotalCost = self.calc_d_total_cost(dS_agg, S, v)

        return CostVariables(TotalCost, dTotalCost)


