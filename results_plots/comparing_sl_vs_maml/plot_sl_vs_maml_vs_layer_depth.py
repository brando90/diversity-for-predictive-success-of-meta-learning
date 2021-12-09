#%%
"""
1. Main goal: compare representations/feature extractor
    - SL vs MAML vs MAML(MAML)

"""
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from uutils.plot import plot_with_error_bands, save_to_desktop

ylim = (-0.05, 1.05)
feature_layers_only: bool = True  # do not set to false, instead check other file

# -- svcca

metric = 'svcca'

# - d(f_sl, A(f_maml)
mus1 = OrderedDict([('model.features.conv1', 0.048231109976768494),
             ('model.features.conv2', 0.37319329380989075),
             ('model.features.conv3', 0.5087650418281555),
             ('model.features.conv4', 0.550204873085022),
             ('model.cls', 0.036156266927719116)])
stds1 = OrderedDict([('model.features.conv1', 0.01699242554605007),
             ('model.features.conv2', 0.008074474520981312),
             ('model.features.conv3', 0.0074108378030359745),
             ('model.features.conv4', 0.011168122291564941),
             ('model.cls', 0.008653730154037476)])
# - d(f_sl, f_maml)
mus2 = OrderedDict([('model.features.conv1', 0.041828498244285583),
             ('model.features.conv2', 0.3761780858039856),
             ('model.features.conv3', 0.5071989297866821),
             ('model.features.conv4', 0.5455787181854248),
             ('model.cls', 0.060455046594142914)])
stds2 = OrderedDict([('model.features.conv1', 0.011576720513403416),
             ('model.features.conv2', 0.007901872508227825),
             ('model.features.conv3', 0.00838900450617075),
             ('model.features.conv4', 0.009392027743160725),
             ('model.cls', 0.01603841595351696)])
# - d(f_maml, A(f_maml))
mus3 = OrderedDict([('model.features.conv1', 6.520509487017989e-05),
             ('model.features.conv2', 0.0031786561012268066),
             ('model.features.conv3', 0.00827802624553442),
             ('model.features.conv4', 0.016165409237146378),
             ('model.cls', 0.5147485733032227)])
stds3 = OrderedDict([('model.features.conv1', 4.952655581291765e-05),
             ('model.features.conv2', 0.004382042214274406),
             ('model.features.conv3', 0.00791084673255682),
             ('model.features.conv4', 0.003718515858054161),
             ('model.cls', 0.04741168022155762)])

x = list(range(1, len(mus1)+1))
x_vals_as_symbols = [layer_name.split('.')[-1].capitalize() for layer_name in mus1.keys()]
x_vals_as_symbols[-1] = 'Head'
title: str = f'Representation distance with metric {metric.upper()}'

if feature_layers_only:
    x = x[:-1]
    x_vals_as_symbols = x_vals_as_symbols[:-1]
    mus1 = list(mus1.values())[:-1]
    stds1 = list(stds1.values())[:-1]
    mus2 = list(mus2.values())[:-1]
    stds2 = list(stds2.values())[:-1]
    mus3 = list(mus3.values())[:-1]
    stds3 = list(stds3.values())[:-1]
    title: str = f'Representation distance with metric {metric.upper()} (feature layer only)'
else:
    mus1 = list(mus1.values())
    stds1 = list(stds1.values())
    mus2 = list(mus2.values())
    stds2 = list(stds2.values())
    mus3 = list(mus3.values())
    stds3 = list(stds3.values())


# - d(f_sl, A(f_maml)
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus1)),
                      yerr=np.asarray(list(stds1)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL vs Adapted MAML', ylim=ylim)
# - d(f_sl, f_maml)
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus2)),
                      yerr=np.asarray(list(stds2)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL vs MAML', ylim=ylim)
# - d(f_maml, A(f_maml))
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus3)),
                      yerr=np.asarray(list(stds3)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='MAML vs Adapted MAML', ylim=ylim)


save_to_desktop(plot_name=f'sl_vs_maml_{metric}_{feature_layers_only=}')
plt.show()
# plt.clf()
# plt.close()

# -- pwcca

metric = 'pwcca'

# - d(f_sl, A(f_maml)
mus1 = OrderedDict([('model.features.conv1', 3.194809039541724e-07),
             ('model.features.conv2', 0.26834550499916077),
             ('model.features.conv3', 0.4336632490158081),
             ('model.features.conv4', 0.5035722255706787),
             ('model.cls', -5.960464477539063e-08)])
stds1 = OrderedDict([('model.features.conv1', 1.448609481258245e-07),
             ('model.features.conv2', 0.008444862440228462),
             ('model.features.conv3', 0.008219581097364426),
             ('model.features.conv4', 0.013965236954391003),
             ('model.cls', 1.5098535754987097e-07)])
# - d(f_sl, f_maml)
mus2 = OrderedDict([('model.features.conv1', 2.455711296533991e-07),
             ('model.features.conv2', 0.2716435194015503),
             ('model.features.conv3', 0.42874592542648315),
             ('model.features.conv4', 0.5053762793540955),
             ('model.cls', 2.38418573772492e-09)])
stds2 = OrderedDict([('model.features.conv1', 1.9941592199756997e-07),
             ('model.features.conv2', 0.007583539932966232),
             ('model.features.conv3', 0.013171483762562275),
             ('model.features.conv4', 0.011725146323442459),
             ('model.cls', 1.2816153116546047e-07)])
# - d(f_maml, A(f_maml))
mus3 = OrderedDict([('model.features.conv1', 1.4781952017983713e-07),
             ('model.features.conv2', 0.0010695743840187788),
             ('model.features.conv3', 0.002788600977510214),
             ('model.features.conv4', 0.010854508727788925),
             ('model.cls', 0.4174900949001312)])
stds3 = OrderedDict([('model.features.conv1', 1.1428974033833583e-07),
             ('model.features.conv2', 0.0004304341855458915),
             ('model.features.conv3', 0.0008126013563014567),
             ('model.features.conv4', 0.0024617069866508245),
             ('model.cls', 0.040015991777181625)])

x = list(range(1, len(mus1)+1))
x_vals_as_symbols = [layer_name.split('.')[-1].capitalize() for layer_name in mus1.keys()]
x_vals_as_symbols[-1] = 'Head'
title: str = f'Representation distance with metric {metric.upper()}'

if feature_layers_only:
    x = x[:-1]
    x_vals_as_symbols = x_vals_as_symbols[:-1]
    mus1 = list(mus1.values())[:-1]
    stds1 = list(stds1.values())[:-1]
    mus2 = list(mus2.values())[:-1]
    stds2 = list(stds2.values())[:-1]
    mus3 = list(mus3.values())[:-1]
    stds3 = list(stds3.values())[:-1]
    title: str = f'Representation distance with metric {metric.upper()} (feature layer only)'
else:
    mus1 = list(mus1.values())
    stds1 = list(stds1.values())
    mus2 = list(mus2.values())
    stds2 = list(stds2.values())
    mus3 = list(mus3.values())
    stds3 = list(stds3.values())

# - d(f_sl, A(f_maml)
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus1)),
                      yerr=np.asarray(list(stds1)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL vs Adapted MAML', ylim=ylim)
# - d(f_sl, f_maml)
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus2)),
                      yerr=np.asarray(list(stds2)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL vs MAML', ylim=ylim)
# - d(f_maml, A(f_maml))
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus3)),
                      yerr=np.asarray(list(stds3)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='MAML vs Adapted MAML', ylim=ylim)


save_to_desktop(plot_name=f'sl_vs_maml_{metric}_{feature_layers_only=}')
plt.show()
# plt.clf()
# plt.close()


# -- lincka

metric = 'lincka'

# - d(f_sl, A(f_maml)
mus1 = OrderedDict([('model.features.conv1', 0.20091617107391357),
             ('model.features.conv2', 0.34751492738723755),
             ('model.features.conv3', 0.7656251788139343),
             ('model.features.conv4', 0.6978272199630737),
             ('model.cls', 0.3908143639564514)])
stds1 = OrderedDict([('model.features.conv1', 0.023728322237730026),
             ('model.features.conv2', 0.026304269209504128),
             ('model.features.conv3', 0.03485113009810448),
             ('model.features.conv4', 0.04014826565980911),
             ('model.cls', 0.09281536191701889)])
# - d(f_sl, f_maml)
mus2 = OrderedDict([('model.features.conv1', 0.2074851095676422),
             ('model.features.conv2', 0.3558051288127899),
             ('model.features.conv3', 0.7673287391662598),
             ('model.features.conv4', 0.677351713180542),
             ('model.cls', 0.8115940690040588)])
stds2 = OrderedDict([('model.features.conv1', 0.03713902831077576),
             ('model.features.conv2', 0.028248080983757973),
             ('model.features.conv3', 0.02466030977666378),
             ('model.features.conv4', 0.030132004991173744),
             ('model.cls', 0.06972156465053558)])
# - d(f_maml, A(f_maml))
mus3 = OrderedDict([('model.features.conv1', 0.0002118802076438442),
             ('model.features.conv2', 0.0010505079990252852),
             ('model.features.conv3', 0.0017140889540314674),
             ('model.features.conv4', 0.031071415171027184),
             ('model.cls', 0.7015092968940735)])
stds3 = OrderedDict([('model.features.conv1', 0.000164504031999968),
             ('model.features.conv2', 0.00037652996252290905),
             ('model.features.conv3', 0.00031865437631495297),
             ('model.features.conv4', 0.005157057661563158),
             ('model.cls', 0.07916036993265152)])

x = list(range(1, len(mus1)+1))
x_vals_as_symbols = [layer_name.split('.')[-1].capitalize() for layer_name in mus1.keys()]
x_vals_as_symbols[-1] = 'Head'
title: str = f'Representation distance with metric {metric.upper()}'

if feature_layers_only:
    x = x[:-1]
    x_vals_as_symbols = x_vals_as_symbols[:-1]
    mus1 = list(mus1.values())[:-1]
    stds1 = list(stds1.values())[:-1]
    mus2 = list(mus2.values())[:-1]
    stds2 = list(stds2.values())[:-1]
    mus3 = list(mus3.values())[:-1]
    stds3 = list(stds3.values())[:-1]
    title: str = f'Representation distance with metric {metric.upper()} (feature layer only)'
else:
    mus1 = list(mus1.values())
    stds1 = list(stds1.values())
    mus2 = list(mus2.values())
    stds2 = list(stds2.values())
    mus3 = list(mus3.values())
    stds3 = list(stds3.values())

# - d(f_sl, A(f_maml)
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus1)),
                      yerr=np.asarray(list(stds1)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL vs Adapted MAML', ylim=ylim)
# - d(f_sl, f_maml)
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus2)),
                      yerr=np.asarray(list(stds2)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL vs MAML', ylim=ylim)
# - d(f_maml, A(f_maml))
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus3)),
                      yerr=np.asarray(list(stds3)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='MAML vs Adapted MAML', ylim=ylim)


save_to_desktop(plot_name=f'sl_vs_maml_{metric}_{feature_layers_only=}')
plt.show()
# plt.clf()
# plt.close()


# -- opd

metric = 'opd'

# - d(f_sl, A(f_maml)
mus1 = OrderedDict([('model.features.conv1', 0.11651492863893509),
             ('model.features.conv2', 0.27015742659568787),
             ('model.features.conv3', 0.491171270608902),
             ('model.features.conv4', 0.5260140299797058),
             ('model.cls', 0.37993162870407104)])
stds1 = OrderedDict([('model.features.conv1', 0.0060066902078688145),
             ('model.features.conv2', 0.009168228134512901),
             ('model.features.conv3', 0.014589996077120304),
             ('model.features.conv4', 0.023802923038601875),
             ('model.cls', 0.05325770005583763)])
# - d(f_sl, f_maml)
mus2 = OrderedDict([('model.features.conv1', 0.1156502515077591),
             ('model.features.conv2', 0.2719881534576416),
             ('model.features.conv3', 0.4852330684661865),
             ('model.features.conv4', 0.5160539150238037),
             ('model.cls', 0.620363712310791)])
stds2 = OrderedDict([('model.features.conv1', 0.0072618694975972176),
             ('model.features.conv2', 0.00974828377366066),
             ('model.features.conv3', 0.009769530966877937),
             ('model.features.conv4', 0.022249022498726845),
             ('model.cls', 0.05263911187648773)])
# - d(f_maml, A(f_maml))
mus3 = OrderedDict([('model.features.conv1', 6.68120410409756e-05),
             ('model.features.conv2', 0.0009591174311935902),
             ('model.features.conv3', 0.0021882248111069202),
             ('model.features.conv4', 0.013909688219428062),
             ('model.cls', 0.47748175263404846)])
stds3 = OrderedDict([('model.features.conv1', 3.6861063563264906e-05),
             ('model.features.conv2', 0.0003826199972536415),
             ('model.features.conv3', 0.0005490550538524985),
             ('model.features.conv4', 0.002069885842502117),
             ('model.cls', 0.04107050225138664)])

x = list(range(1, len(mus1)+1))
x_vals_as_symbols = [layer_name.split('.')[-1].capitalize() for layer_name in mus1.keys()]
x_vals_as_symbols[-1] = 'Head'
title: str = f'Representation distance with metric {metric.upper()}'

if feature_layers_only:
    x = x[:-1]
    x_vals_as_symbols = x_vals_as_symbols[:-1]
    mus1 = list(mus1.values())[:-1]
    stds1 = list(stds1.values())[:-1]
    mus2 = list(mus2.values())[:-1]
    stds2 = list(stds2.values())[:-1]
    mus3 = list(mus3.values())[:-1]
    stds3 = list(stds3.values())[:-1]
    title: str = f'Representation distance with metric {metric.upper()} (feature layer only)'
else:
    mus1 = list(mus1.values())
    stds1 = list(stds1.values())
    mus2 = list(mus2.values())
    stds2 = list(stds2.values())
    mus3 = list(mus3.values())
    stds3 = list(stds3.values())

# - d(f_sl, A(f_maml)
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus1)),
                      yerr=np.asarray(list(stds1)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL vs Adapted MAML', ylim=ylim)
# - d(f_sl, f_maml)
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus2)),
                      yerr=np.asarray(list(stds2)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL vs MAML', ylim=ylim)
# - d(f_maml, A(f_maml))
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus3)),
                      yerr=np.asarray(list(stds3)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='MAML vs Adapted MAML', ylim=ylim)


save_to_desktop(plot_name=f'sl_vs_maml_{metric}_{feature_layers_only=}')
plt.show()
# plt.clf()
# plt.close()