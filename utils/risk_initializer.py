import numpy as np
from numpy.typing import NDArray


class RiskInitializer:
    def __init__(self, max_ratio=3000, max_base=1000, exponent=6, seed=0):
        self.rnd = np.random.default_rng(seed)
        self.max_ratio = max_ratio
        self.max_base = max_base
        self.exponent = exponent

    def age_to_risk_exponential_func(self, age: NDArray[float], size=1) -> NDArray[float]:
        base = self.rnd.uniform(0, self.max_base, size=(size, 1))
        ratio = self.rnd.uniform(0, self.max_ratio, size=(size, 1))
        return base * ((ratio - 1) * (age / 100) ** self.exponent + 1)

