import pandas as pd
import numpy as np


def parse_var(row, names, var):
    flat = row[var].flatten()
    for i, name in enumerate(names):
        row[names] = flat[i]


def parse_col(df, col_name, groups, suffixes_list=None):
    if suffixes_list is None:
        suffixes_list = range(1, groups + 1)
    elif len(suffixes_list) != groups:
        raise ValueError('groups and name are not of the same length')
    names = [f'{col_name}_{suf}' for suf in suffixes_list]
    df[names] = pd.DataFrame(df[col_name].apply(np.ravel, order='F').to_list())


def parse_test(path, groups, extra_cols_to_parse=[]):
    base = pd.read_pickle(path)

    for col in ['sol', 'sol_gov']:
        base = pd.concat([base.drop(col, axis=1), pd.json_normalize(base[col]).add_suffix('_' + col)], axis=1)

    for col in ['cost_sol', 'cost_sol_gov']:
        base[col] = base[col].apply(lambda x: x.flatten())
        base[col] = pd.DataFrame(base[col].to_list()).sum(axis=1)

    base['sol_gap'] = base['cost_sol'] - base['cost_sol_gov']

    best_sol_indx = np.argmin(base[['cost_sol', 'cost_sol_gov']].to_numpy(), axis=1)
    base['best_sol'] = np.take(['cost_sol', 'cost_sol_gov'], best_sol_indx)

    for col in ['risk_l', 'v_sol', 'S_sol', 'S_sol_gov'] + extra_cols_to_parse:
        parse_col(base, col, groups)

    return base


def rename_cols(df, factor, group_to_compare):
    df = df.copy()

    max_l_to_compare = df['risk_l_1'].max() * factor
    df.query(f'risk_l_{group_to_compare} < {max_l_to_compare}', inplace=True)

    df['best_sol'] = df['best_sol'].str.replace('cost_sol_gov', 'government')
    df['best_sol'] = df['best_sol'].str.replace('cost_sol', 'equilibrium')
    df.rename({'best_sol': 'Lowest Cost', 'risk_l_1': 'Young Cost of Infection', f'risk_l_{group_to_compare}': 'Old Cost of Infection'}, axis=1, inplace=True)
    df['factor'] = (df['Young Cost of Infection'] * factor)
    return df