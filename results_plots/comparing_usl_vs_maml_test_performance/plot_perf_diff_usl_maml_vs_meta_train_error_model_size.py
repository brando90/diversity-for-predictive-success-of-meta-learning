"""
Report: https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/reports/Is-the-size-of-the-model-playing-a-key-role-in-meta-learning---VmlldzoxOTM0NDk5

mainly to check if diff = acc(maml) - acc(usl) variaes wrt model size/meta-train error.
hypothesis: is that smaller models need feature extractor to be adapted more (meta-learning matters more).
Answer: TODO
"""

#%%
from matplotlib import pyplot as plt

from uutils.plot import plot_with_error_bands, save_to_desktop
import numpy as np

"""
5CNN, on MI
"""


# -- filter size is the number of different filters per layer filter_size (=out_channels, Number of channels produced by the convolution) is changing, so the size of the tensor/box. Each channel corresponds to 1 filter, so one feature detactor (that is shared spatially)
filter_size_per_layer = [16, 32]

maml5_test_acc = [0.5448923252224922, 0.5421846333742142]
maml5_test_acc_ci = [0.00776597815176418, 0.008195475209108384]

maml10_test_acc = [0.542400018543005, 0.543046172440052]
maml10_test_acc_ci = [0.007827878078656139, 0.007480880752067012]

usl_test_acc = [0.5422461538461539, 0.5587692307692307]
usl_test_acc_ci = [0.00736747015201177, 0.007960446033549236]

# test_performance_diff = np.array(maml5_test_acc) - np.array(usl_test_acc)
# test_performance_diff = np.array(maml10_test_acc) - np.array(usl_test_acc)
# test_performance_diff = [0.0, 0.5421846333742142+0.008195475209108384 - (0.5587692307692307-0.007960446033549236)]

# - x-axis: "model size" e.g. hidden units, number weights, depth, meta-train error
# x_axis = []
x_axis = filter_size_per_layer

# - y-axis: diff = acc(maml) - acc(usl)
y_axis = []

# - plot it
ylim = (0.5, 0.6)
# ylim = None
title = 'Model Size Effect in Performance Difference of USL vs MAML'
plot_with_error_bands(x=filter_size_per_layer,
                      y=np.asarray(list(maml5_test_acc)),

                      yerr=np.asarray(list(maml5_test_acc_ci)),

                      xlabel='filter size', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='MAML5 Acc', ylim=ylim)

plot_with_error_bands(x=filter_size_per_layer,
                      y=np.asarray(list(usl_test_acc)),

                      yerr=np.asarray(list(usl_test_acc_ci)),

                      xlabel='filter size', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='USL Acc', ylim=ylim)


save_to_desktop(plot_name=f'diff_maml_usl_vs_filter_size')
plt.show()

#%%