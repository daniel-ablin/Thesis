import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple


def update_v(v: NDArray[float], grad: NDArray[float], learning_rate: NDArray[float], epsilon: float) -> NDArray[float]:
    decent = grad * learning_rate
    decent = np.minimum(abs(decent), 0.01) * np.sign(decent)

    v_new = v - decent
    v_new = np.minimum(np.maximum(v_new, epsilon), 1)

    return v_new


def break_condition_test(dTotalCost: NDArray[float], itr: int, stop_itr: int, threshold: float, grad: NDArray[float], epsilon: float,
                         v: NDArray[float]) -> Tuple[bool, str]:
    if not itr == 0 and itr % stop_itr == 0:
        if (abs((dTotalCost[itr - stop_itr - 1:itr - 1].sum(axis=0) - dTotalCost[itr] * stop_itr)) < threshold).all():
            if (abs(grad) < threshold).all():
                msg = 'found solution'
                return True, msg
            elif ((v == 1) * (dTotalCost[itr] < 0)).any() or (
                    (v == epsilon) * (dTotalCost[itr] > 0)).any():
                msg = 'no close solution'
                return True, msg
    msg = 'time out'
    return False, msg


def get_d_matrix(groups: int, norm_to_one_meeting: bool = False) -> Tuple[NDArray[float], NDArray[float], float]:
    base_d = pd.read_csv('d_params.csv', header=None).to_numpy()
    age_groups = np.arange(5, 85, 5)
    if groups == 2:
        groups = [10]
    if isinstance(groups, list):
        d_row_split = np.split(base_d, groups)
        d_full_split = [np.split(row_split, groups, axis=1) for row_split in d_row_split]
        split_age = np.split(age_groups, groups)
        d = np.array([[split.sum(axis=0).mean() for split in row_split] for row_split in d_full_split]).T
    else:
        d_row_split = np.array_split(base_d, groups)
        d_full_split = [np.array_split(row_split, groups, axis=1) for row_split in d_row_split]
        split_age = np.array_split(age_groups, groups)
        d = np.array([[split.sum(axis=0).mean() for split in row_split] for row_split in d_full_split]).T

    mean_age = np.array([np.insert(split, 0, split[0] - 5).mean() for split in split_age])
    if norm_to_one_meeting:
        norm_factor = d.sum(axis=1).max()
    else:
        norm_factor = 1
    d /= norm_factor
    return d, mean_age, norm_factor


def refactor_d_for_SI_simulation(d, fact=2) -> Tuple[NDArray[float], NDArray[float]]:
    d_base = d.sum(axis=1) / fact  # np.ones(d.shape[0])
    d.fill(0)
    np.fill_diagonal(d, d_base)
    d_update_rule = np.ones(d.shape) * d_base
    np.fill_diagonal(d_update_rule, -d_base)
    return d, d_update_rule


def calc_diag(p: int, i: int, w: int) -> Tuple[int, int]:
    if p == i and p == w:
        return 2, p
    elif p != i and i != w:
        return 0, 0
    elif p == i and p != w:
        return 1, w
    else:
        return 1, p


def get_populations_proportions(d: NDArray[float]) -> NDArray[float]:
    return np.nan_to_num((d[0, :] / d[:, 0]), nan=0).reshape(-1, 1)


def norm_d_to_one(d: NDArray[float], beta: float) -> Tuple[NDArray[float], NDArray[float]]:
    factor = d.sum(axis=1)
    d = d/factor[:, np.newaxis]
    beta = beta*factor
    return d, beta
