

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
ylim = (0.2, 1)
xlim= (0,0.575)
#Low diversity regime
div = np.array([0.247,0.271,0.393,0.47])[:3]
maml5 = np.array([0.2019066676,0.3696800001,0.5679199995,0.8206266673])[:3]
maml5e = np.array([0.002677290758,0.00682331957,0.009613171253,0.01028397098])[:3]
maml10 = np.array([0.2014800006,0.3736666677,0.5706933325,0.8358133346])[:3]
maml10e = np.array([0.002770970995,0.007154970165,0.00928642408,0.009982345466])[:3]
usl = np.array([0.2001733333,0.3775733333,0.57124,0.8167066667])[:3]
usle = np.array([0.0006564367522,0.006937149032,0.009459490881,0.009160933492])[:3]
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
