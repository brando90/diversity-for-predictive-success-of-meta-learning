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


#%%
"""
alla archs, all models
"""

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]
ax.bar(langs,students)
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
data = [[30, 25, 50, 20],
[40, 23, 51, 17],
[35, 22, 45, 19]]
X = np.arange(4)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, data[0], color = 'b', width = 0.25)
ax.bar(X + 0.25, data[1], color = 'g', width = 0.25)
ax.bar(X + 0.50, data[2], color = 'r', width = 0.25)


