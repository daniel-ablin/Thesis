from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


class ModelVariable:
    def __init__(self, anarchy, gov=None):
        self.anarchy = anarchy
        if gov:
            self.gov = gov
        else:
            self.gov = deepcopy(anarchy)

    def __getitem__(self, item: int) -> 'ModelVariable':
        return ModelVariable(self.anarchy[item], self.gov[item])

    def __setitem__(self, key: int, value: 'ModelVariable'):
        self.gov[key] = value.gov
        self.anarchy[key] = value.anarchy

    def restart_variable(self, starting_value=0, starting_index=0):
        self.anarchy[starting_index] = starting_value
        self.gov[starting_index] = starting_value

    @staticmethod
    def activate_function(func, *args) -> 'ModelVariable':
        anarchy_args = (arg.anarchy for arg in args)
        gov_args = (arg.gov for arg in args)
        return ModelVariable(func(*anarchy_args), func(*gov_args))


@dataclass
class DynamicsVariables:
    I: np.ndarray
    S: np.ndarray
    dS: np.ndarray
    dI: np.ndarray

    def __getitem__(self, item: int) -> 'DynamicsVariables':
        return DynamicsVariables(I=self.I[item], S=self.S[item], dS=self.dS[item], dI=self.dI[item])


@dataclass
class CostVariables:
    TotalCost: np.ndarray
    dTotalCost: np.ndarray

    def __getitem__(self, item: int) -> 'CostVariables':
        return CostVariables(TotalCost=self.TotalCost[item], dTotalCost=self.dTotalCost[item])

    def __setitem__(self, key: int, value: 'CostVariables'):
        self.TotalCost[key] = value.TotalCost
        self.dTotalCost[key] = value.dTotalCost


@dataclass
class ModelOuterVariables:
    groups: int
    T: int
    beta: NDArray[float]
    recovered_rate: float
    I0: float
    elasticity_adjust: float
    d: NDArray[float]
    populations_proportions: NDArray[float]
    risk_l: NDArray[float]
