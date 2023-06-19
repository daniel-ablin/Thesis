import seaborn as sns
import matplotlib.pyplot as plt

import sys

try:
    del sys.modules['results_analisys.utils']
except:
    pass

from results_analisys.utils import *

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
sns.set(font_scale=2)

specific_file = '/Users/danielablin/Documents/GitHub/Thesis/test_run_100_5_2023-03-12 12:57:31.251713.pickle'
base = parse_test(specific_file, 5)

fig, ax = plt.subplots(2)
fig.set_figheight(12)
fig.set_figwidth(15)

test = rename_cols(base, 39, 5)
test['Oldest vs Youngest Cost of Infection Rate'] = test['Old Cost of Infection'] / test['Young Cost of Infection']
test.rename({'risk_l_2': 'Second Group Cost of Infection'}, axis=1, inplace=True)
test['line'] = 50
test['line2'] = 39

sns.scatterplot(data=test, x='Second Group Cost of Infection', y='Oldest vs Youngest Cost of Infection Rate', hue='Lowest Cost', hue_order=['equilibrium', 'government'], ax=ax[0])
sns.scatterplot(data=test, x='Young Cost of Infection', y='Oldest vs Youngest Cost of Infection Rate', hue='Lowest Cost', hue_order=['equilibrium', 'government'], ax=ax[1])
# sns.scatterplot(data=test, x='line', y='Oldest vs Youngest Cost of Infection Rate', color='green', ax=ax[1])
# sns.scatterplot(data=test, x='Young Cost of Infection', y='line2', color='green', ax=ax[1])
ax[1].set_ylim(0, 150)

plt.show()
