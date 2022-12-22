

# %%
from matplotlib import pyplot as plt

from uutils.plot import plot_with_error_bands, save_to_desktop
import numpy as np

"""
5CNN, on MI
"""

# - x-axis: "model size" e.g. hidden units, number weights, depth, meta-train error
# x_axis = []
#x_axis = []

# - y-axis: diff = acc(maml) - acc(usl)
#y_axis = []

# - plot it
xlim = (0.18, 0.26)
ylim= (0.5,1)

#
#div = np.array([3.58E-05,0.1833,0.5686,0.86])[:3]
#maml5 = np.array([0.2019066676,0.3696800001,0.5679199995,0.8206266673])[:3]
#maml5e = np.array([0.002677290758,0.00682331957,0.009613171253,0.01028397098])[:3]
#maml10 = np.array([0.2014800006,0.3736666677,0.5706933325,0.8358133346])[:3]
#maml10e = np.array([0.002770970995,0.007154970165,0.00928642408,0.009982345466])[:3]
#usl = np.array([0.2001733333,0.3775733333,0.57124,0.8167066667])[:3]
#usle = np.array([0.0006564367522,0.006937149032,0.009459490881,0.009160933492])[:3]

div = np.array([0.187,0.207,0.225,0.232,0.253])
maml5 = np.array([0.8581999791,0.5979999825,0.856199978,0.6979999807,0.8693999767])
maml5e = np.array([0.02028297981,0.01557530173,0.01602032096,0.03264911088,0.01528618595])
maml10= np.array([0.8571999764,0.5729999858,0.8687999761,0.6913999826,0.8765999788])
maml10e =np.array([0.01799677471,0.01552287091,0.01567527492,0.0308749259,0.01631899972])
usl5= np.array([0.7819999781,0.5399999863,0.81279998,0.7125999826,0.9189999789])
usl5e= np.array([0.01848926904,0.01675138962,0.02008139807,0.02777992379,0.01521233022])
usl10= np.array([0.78279998,0.5429999852,0.8193999791,0.7111999831,0.9199999797])
usl10e=np.array([0.016996685,0.01865095678,0.01865163857,0.02260646053,0.01527103805])



# ylim = None
title = 'Diversity Effect in Performance Difference of USL vs MAML'
plot_with_error_bands(x=div,
                      y=np.asarray(list(maml5)),

                      yerr=np.asarray(list(maml5e)),

                      xlabel='Hellinger Diversity Coefficient', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='MAML5 Acc', ylim=ylim,xlim=xlim)
plot_with_error_bands(x=div,
                      y=np.asarray(list(usl5)),

                      yerr=np.asarray(list(usl5e)),

                      xlabel='Hellinger Diversity Coefficient', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='USL Acc', ylim=ylim,xlim=xlim)
save_to_desktop(plot_name=f'diff_maml5_usl_vs_hellinger')
#plt.show()

plot_with_error_bands(x=div,
                      y=np.asarray(list(maml10)),

                      yerr=np.asarray(list(maml10e)),

                      xlabel='Hellinger Diversity Coefficient', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='MAML10 Acc', ylim=ylim,xlim=xlim)

plot_with_error_bands(x=div,
                      y=np.asarray(list(usl10)),

                      yerr=np.asarray(list(usl10e)),

                      xlabel='Hellinger Diversity Coefficient', ylabel=f'Meta-Test Accuracy',
                      title=title, curve_label='USL Acc', ylim=ylim,xlim=xlim)
save_to_desktop(plot_name=f'diff_maml10_usl_vs_hellinger')
#plt.show()

