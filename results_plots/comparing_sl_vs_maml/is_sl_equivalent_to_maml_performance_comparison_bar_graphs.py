# %%
"""
plot (maml5, maml5), (maml5, maml), (USL, LR) with error bars to see if the CI intersect.
"""

# - plot bands
from matplotlib import pyplot as plt

# x = [1]
# y = [62.4]
# yerr = [1.64]
# plt.errorbar(x=x, y=y, yerr=yerr, color='blue')
# # x = [1]
# # y = [62.3]
# # yerr = [1.50]
# # plt.errorbar(x=x, y=y, yerr=yerr, color='red')
# x = [3]
# y = [60.1]
# yerr = [1.37]
# plt.errorbar(x=x, y=y, yerr=yerr, color='red')


import matplotlib.pyplot as plt

fig = plt.figure()
# ax = fig.add_axes([0, 0, 1])
adapted_models = ['MAML5', 'MAML10', 'USL (TL)']
meta_test_acc = [62.4, 62.3, 60.1]
meta_test_ci = [1.64, 1.5, 1.37]
ax.bar(adapted_models, meta_test_acc)
plt.show()

# %%
"""
5CNN MI
"""

import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
ind = np.arange(N)  # the x locations for the groups
width = 0.35
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(ind, menMeans, width, color='r')
ax.bar(ind, womenMeans, width, bottom=menMeans, color='b')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
ax.set_yticks(np.arange(0, 81, 10))
ax.legend(labels=['Men', 'Women'])
plt.show()

# %%
"""
resnet12 MI
"""

# %%
"""
alla archs, all models
"""

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23, 17, 35, 29, 12]
ax.bar(langs, students)
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt

data = [[30, 25, 50, 20],
        [40, 23, 51, 17],
        [35, 22, 45, 19]]
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
ax.bar(X + 0.00, data[0], color='b', width=0.25)
ax.bar(X + 0.25, data[1], color='g', width=0.25)
ax.bar(X + 0.50, data[2], color='r', width=0.25)

# %%

import pandas as pd

speed = [0.1, 17.5, 40, 48, 52, 69, 88]
lifespan = [2, 8, 70, 1.5, 25, 12, 28]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']
df = pd.DataFrame({'speed': speed,
                   'lifespan': lifespan}, index=index)
print(df)
ax = df.plot.bar(rot=0)

plt.show()

# %%
import pandas as pd

speed = [0.1, 17.5, 40, 48, 52, 69, 88]
index = ['snail', 'pig', 'elephant',
         'rabbit', 'giraffe', 'coyote', 'horse']
df = pd.DataFrame({'speed': speed,
                   }, index=index)
print(df)
ax = df.plot.bar(rot=0)

plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt
from numpy import array

df = pd.DataFrame([[4, 6, 1, 3],
                   [5, 7, 5, 2]],
                  columns=['mean1', 'mean2', 'std1', 'std2'], index=['A', 'B'])
print(df)

# convert the std columns to an array
yerr = df[['std1', 'std2']].to_numpy().T

df[['mean1', 'mean2']].plot(kind='bar', yerr=yerr, alpha=0.5, error_kw=dict(ecolor='k'), capsize=5.0)
plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt

groups = ['MI 5CNN']
adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']
meta_test_acc = [62.4, 62.3, 60.1]
meta_test_ci = [1.64, 1.5, 1.37]
row1 = array([meta_test_acc + meta_test_ci])
data = row1

df = pd.DataFrame(data, columns=adapted_models, index=groups)
print(df)

# convert the std columns to an array
yerr = df[['MAML5 ci', 'MAML10 ci', 'USL ci']].to_numpy().T
print(yerr)

# df[['MAML5', 'MAML10', 'USL']].plot(kind='bar', yerr=yerr, alpha=0.5, error_kw=dict(ecolor='k'), capsize=5.0)
df[['MAML5', 'MAML10', 'USL']].plot(kind='bar', yerr=yerr, alpha=0.7, capsize=5.0, width=0.08)
# plt.grid(True)
plt.grid(linestyle='--')
plt.tight_layout()
plt.xticks(rotation=0)
plt.show()

# %%

import pandas as pd
import matplotlib.pyplot as plt

groups = ['MI 5CNN', 'MI 5CNN 2']  # the rows of a df
adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
meta_test_acc = [62.4, 62.3, 60.1]
meta_test_ci = [1.64, 1.5, 1.37]
row1 = meta_test_acc + meta_test_ci
row2 = meta_test_acc + meta_test_ci
data = [row1, row2]
print(data)

df = pd.DataFrame(data, columns=adapted_models, index=groups)
print(df)

# convert the std columns to an array
yerr = df[['MAML5 ci', 'MAML10 ci', 'USL ci']].to_numpy().T
print(f'{yerr=}')

# df[['MAML5', 'MAML10', 'USL']].plot(kind='bar', yerr=yerr, alpha=0.5, error_kw=dict(ecolor='k'), capsize=5.0)
df[['MAML5', 'MAML10', 'USL']].plot(kind='bar', yerr=yerr, alpha=0.7, capsize=2.5, width=0.15)
# plt.grid(True)
plt.grid(linestyle='--')
plt.tight_layout()
plt.xticks(rotation=0)
plt.show()
