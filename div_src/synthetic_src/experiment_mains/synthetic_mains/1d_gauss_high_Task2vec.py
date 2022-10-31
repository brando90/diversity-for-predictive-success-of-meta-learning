

# %%
from matplotlib import pyplot as plt

from uutils.plot import plot_with_error_bands, save_to_desktop
import numpy as np

"""
5CNN, on MI
"""



# test_performance_diff = np.array(maml5_test_acc) - np.array(usl_test_acc)
# test_performance_diff = np.array(maml10_test_acc) - np.array(usl_test_acc)
# test_performance_diff = [0.0, 0.5421846333742142+0.008195475209108384 - (0.5587692307692307-0.007960446033549236)]

# - x-axis: "model size" e.g. hidden units, number weights, depth, meta-train error
# x_axis = []
x_axis = filter_size_per_layer

# - y-axis: diff = acc(maml) - acc(usl)
y_axis = []

# - plot it
ylim = (0.55, 1)
xlim= (0.555,1)
#Low diversity regime
#High diversity regime
div = np.array([0.5686,0.86,0.9317,0.9736,0.9984])
maml5 = np.array([0.5679199995,0.8206266673,0.8570666689,0.8745333358,0.9065333388])
maml5e = np.array([0.009613171253,0.01028397098,0.009584906661,0.01035564055,0.01121512625])
maml10 = np.array([0.5706933325,0.8358133346,0.8720800033,0.885480003,0.9108933385])
maml10e = np.array([0.00928642408,0.009982345466,0.009846180448,0.009894487608,0.01062369246])
usl = np.array([0.57124,00.8167066667,0.8963866667,0.9359066667,0.9824533333])
usle = np.array([0.009459490881,0.009160933492,0.008138150048,0.00718109911,0.004205302444])
# ylim = None
title = 'Diversity Effect in Performance Difference of USL vs MAML'
plot_with_error_bands(x=div,
                      y=np.asarray(list(maml5)),

                      yerr=np.asarray(list(maml5e)),

                      xlabel='Task2Vec Diversity Coefficient', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='MAML5 Acc', ylim=ylim,xlim=xlim)
plot_with_error_bands(x=div,
                      y=np.asarray(list(usl)),

                      yerr=np.asarray(list(usle)),

                      xlabel='Task2Vec Diversity Coefficient', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='USL Acc', ylim=ylim,xlim=xlim)
save_to_desktop(plot_name=f'diff_maml5_usl_vs_task2vec')
plt.show()

plot_with_error_bands(x=div,
                      y=np.asarray(list(maml10)),

                      yerr=np.asarray(list(maml10e)),

                      xlabel='Task2Vec Diversity Coefficient', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='MAML10 Acc', ylim=ylim,xlim=xlim)

plot_with_error_bands(x=div,
                      y=np.asarray(list(usl)),

                      yerr=np.asarray(list(usle)),

                      xlabel='Task2Vec Diversity Coefficient', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='USL Acc', ylim=ylim,xlim=xlim)
save_to_desktop(plot_name=f'diff_maml10_usl_vs_task2vec')
plt.show()
