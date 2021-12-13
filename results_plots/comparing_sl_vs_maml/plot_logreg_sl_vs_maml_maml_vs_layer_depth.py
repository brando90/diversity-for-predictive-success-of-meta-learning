#%%
"""
2. LR(f_sl) vs MAML(f_maml) - compare predictor layers
    - goal is to compare what the two base_models do at inference by emphasizing the final layer + showing the rest of the
    layers allows to see a pattern from the rep layer to the final layer.
"""
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt

from uutils.plot import plot_with_error_bands, save_to_desktop

ylim = (-0.05, 1.05)
feature_layers_only: bool = False  # the key is to see the predictor layer difference

# -- svcca

metric = 'svcca'

# - d(f_sl, A(f_maml))
# mus1 = OrderedDict([('model.features.conv1', 0.048231109976768494),
#              ('model.features.conv2', 0.37319329380989075),
#              ('model.features.conv3', 0.5087650418281555),
#              ('model.features.conv4', 0.550204873085022),
#              ('model.cls', 0.036156266927719116)])
# stds1 = OrderedDict([('model.features.conv1', 0.01699242554605007),
#              ('model.features.conv2', 0.008074474520981312),
#              ('model.features.conv3', 0.0074108378030359745),
#              ('model.features.conv4', 0.011168122291564941),
#              ('model.cls', 0.008653730154037476)])
# - d(LR(f_sl), A(f_maml))
mus2 = OrderedDict([('model.features.conv1', 0.048231109976768494),
             ('model.features.conv2', 0.37319329380989075),
             ('model.features.conv3', 0.5087650418281555),
             ('model.features.conv4', 0.550204873085022),
             ('model.cls', 0.3283124268054962)])
stds2 = OrderedDict([('model.features.conv1', 0.01699242554605007),
             ('model.features.conv2', 0.008074474520981312),
             ('model.features.conv3', 0.0074108378030359745),
             ('model.features.conv4', 0.011168122291564941),
             ('model.cls', 0.07408269494771957)])
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

x = list(range(1, len(mus2)+1))
x_vals_as_symbols = [layer_name.split('.')[-1].capitalize() for layer_name in mus2.keys()]
x_vals_as_symbols[-1] = 'Head'
title: str = f'Representation distance with metric {metric.upper()}'

if feature_layers_only:
    x = x[:-1]
    x_vals_as_symbols = x_vals_as_symbols[:-1]
    # mus1 = list(mus1.values())[:-1]
    # stds1 = list(stds1.values())[:-1]
    mus2 = list(mus2.values())[:-1]
    stds2 = list(stds2.values())[:-1]
    mus3 = list(mus3.values())[:-1]
    stds3 = list(stds3.values())[:-1]
    title: str = f'Representation distance with metric {metric.upper()} (feature layer only)'
else:
    # mus1 = list(mus1.values())
    # stds1 = list(stds1.values())
    mus2 = list(mus2.values())
    stds2 = list(stds2.values())
    mus3 = list(mus3.values())
    stds3 = list(stds3.values())


# - d(f_sl, A(f_maml))
# plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus1)),
#                       yerr=np.asarray(list(stds1)),
#                       xlabel='Layer', ylabel=f'Distance {metric.upper()}',
#                       title=title, curve_label='SL vs Adapted MAML', ylim=ylim)
# - d(LR(f_sl), A(f_maml))
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus2)),
                      yerr=np.asarray(list(stds2)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL Adapted Head vs Adapted MAML', ylim=ylim)
# - d(f_maml, A(f_maml))
# plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus3)),
#                       yerr=np.asarray(list(stds3)),
#                       xlabel='Layer', ylabel=f'Distance {metric.upper()}',
#                       title=title, curve_label='MAML vs Adapted MAML', ylim=ylim)


save_to_desktop(plot_name=f'lr_sl_vs_maml_maml_{metric}_{feature_layers_only=}')
plt.show()
# plt.clf()
# plt.close()

# -- pwcca

metric = 'pwcca'

# - d(f_sl, A(f_maml)
# mus1 = OrderedDict([('model.features.conv1', 3.194809039541724e-07),
#              ('model.features.conv2', 0.26834550499916077),
#              ('model.features.conv3', 0.4336632490158081),
#              ('model.features.conv4', 0.5035722255706787),
#              ('model.cls', -5.960464477539063e-08)])
# stds1 = OrderedDict([('model.features.conv1', 1.448609481258245e-07),
#              ('model.features.conv2', 0.008444862440228462),
#              ('model.features.conv3', 0.008219581097364426),
#              ('model.features.conv4', 0.013965236954391003),
#              ('model.cls', 1.5098535754987097e-07)])
# - d(LR(f_sl), A(f_maml))
mus2 = OrderedDict([('model.features.conv1', 3.743171816950053e-07),
             ('model.features.conv2', 0.26837778091430664),
             ('model.features.conv3', 0.4304948151111603),
             ('model.features.conv4', 0.5014489889144897),
             ('model.cls', -7.1525572131747595e-09)])
stds2 = OrderedDict([('model.features.conv1', 1.5051402613153186e-07),
             ('model.features.conv2', 0.010398694314062595),
             ('model.features.conv3', 0.008332903496921062),
             ('model.features.conv4', 0.014489657245576382),
             ('model.cls', 1.3363498396756768e-07)])
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

x = list(range(1, len(mus2)+1))
x_vals_as_symbols = [layer_name.split('.')[-1].capitalize() for layer_name in mus2.keys()]
x_vals_as_symbols[-1] = 'Head'
title: str = f'Representation distance with metric {metric.upper()}'

if feature_layers_only:
    x = x[:-1]
    x_vals_as_symbols = x_vals_as_symbols[:-1]
    # mus1 = list(mus1.values())[:-1]
    # stds1 = list(stds1.values())[:-1]
    mus2 = list(mus2.values())[:-1]
    stds2 = list(stds2.values())[:-1]
    mus3 = list(mus3.values())[:-1]
    stds3 = list(stds3.values())[:-1]
    title: str = f'Representation distance with metric {metric.upper()} (feature layer only)'
else:
    # mus1 = list(mus1.values())
    # stds1 = list(stds1.values())
    mus2 = list(mus2.values())
    stds2 = list(stds2.values())
    mus3 = list(mus3.values())
    stds3 = list(stds3.values())

# - d(f_sl, A(f_maml))
# plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus1)),
#                       yerr=np.asarray(list(stds1)),
#                       xlabel='Layer', ylabel=f'Distance {metric.upper()}',
#                       title=title, curve_label='SL vs Adapted MAML', ylim=ylim)
# - d(LR(f_sl), A(f_maml))
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus2)),
                      yerr=np.asarray(list(stds2)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL Adapted Head vs Adapted MAML', ylim=ylim)
# - d(f_maml, A(f_maml))
# plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus3)),
#                       yerr=np.asarray(list(stds3)),
#                       xlabel='Layer', ylabel=f'Distance {metric.upper()}',
#                       title=title, curve_label='MAML vs Adapted MAML', ylim=ylim)


save_to_desktop(plot_name=f'lr_sl_vs_maml_maml_{metric}_{feature_layers_only=}')
plt.show()
# plt.clf()
# plt.close()


# -- lincka

metric = 'lincka'

# - d(f_sl, A(f_maml))
# mus1 = OrderedDict([('model.features.conv1', 0.20091617107391357),
#              ('model.features.conv2', 0.34751492738723755),
#              ('model.features.conv3', 0.7656251788139343),
#              ('model.features.conv4', 0.6978272199630737),
#              ('model.cls', 0.3908143639564514)])
# stds1 = OrderedDict([('model.features.conv1', 0.023728322237730026),
#              ('model.features.conv2', 0.026304269209504128),
#              ('model.features.conv3', 0.03485113009810448),
#              ('model.features.conv4', 0.04014826565980911),
#              ('model.cls', 0.09281536191701889)])
# - d(LR(f_sl), A(f_maml))
mus2 = OrderedDict([('model.features.conv1', 0.20091617107391357),
             ('model.features.conv2', 0.34751492738723755),
             ('model.features.conv3', 0.7656251788139343),
             ('model.features.conv4', 0.6978272199630737),
             ('model.cls', 0.36451616883277893)])
stds2 = OrderedDict([('model.features.conv1', 0.023728322237730026),
             ('model.features.conv2', 0.026304269209504128),
             ('model.features.conv3', 0.03485113009810448),
             ('model.features.conv4', 0.04014826565980911),
             ('model.cls', 0.11854448914527893)])
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

x = list(range(1, len(mus2)+1))
x_vals_as_symbols = [layer_name.split('.')[-1].capitalize() for layer_name in mus2.keys()]
x_vals_as_symbols[-1] = 'Head'
title: str = f'Representation distance with metric {metric.upper()}'

if feature_layers_only:
    x = x[:-1]
    x_vals_as_symbols = x_vals_as_symbols[:-1]
    # mus1 = list(mus1.values())[:-1]
    # stds1 = list(stds1.values())[:-1]
    mus2 = list(mus2.values())[:-1]
    stds2 = list(stds2.values())[:-1]
    mus3 = list(mus3.values())[:-1]
    stds3 = list(stds3.values())[:-1]
    title: str = f'Representation distance with metric {metric.upper()} (feature layer only)'
else:
    # mus1 = list(mus1.values())
    # stds1 = list(stds1.values())
    mus2 = list(mus2.values())
    stds2 = list(stds2.values())
    mus3 = list(mus3.values())
    stds3 = list(stds3.values())

# - d(f_sl, A(f_maml))
# plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus1)),
#                       yerr=np.asarray(list(stds1)),
#                       xlabel='Layer', ylabel=f'Distance {metric.upper()}',
#                       title=title, curve_label='SL vs Adapted MAML', ylim=ylim)

# - d(LR(f_sl), A(f_maml))
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus2)),
                      yerr=np.asarray(list(stds2)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL Adapted Head vs Adapted MAML', ylim=ylim)
# - d(f_maml, A(f_maml))
# plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus3)),
#                       yerr=np.asarray(list(stds3)),
#                       xlabel='Layer', ylabel=f'Distance {metric.upper()}',
#                       title=title, curve_label='MAML vs Adapted MAML', ylim=ylim)


save_to_desktop(plot_name=f'lr_sl_vs_maml_maml_{metric}_{feature_layers_only=}')
plt.show()
# plt.clf()
# plt.close()


# -- opd

metric = 'opd'

# - d(f_sl, A(f_maml))
# mus1 = OrderedDict([('model.features.conv1', 0.11651492863893509),
#              ('model.features.conv2', 0.27015742659568787),
#              ('model.features.conv3', 0.491171270608902),
#              ('model.features.conv4', 0.5260140299797058),
#              ('model.cls', 0.37993162870407104)])
# stds1 = OrderedDict([('model.features.conv1', 0.0060066902078688145),
#              ('model.features.conv2', 0.009168228134512901),
#              ('model.features.conv3', 0.014589996077120304),
#              ('model.features.conv4', 0.023802923038601875),
#              ('model.cls', 0.05325770005583763)])
# - d(LR(f_sl), A(f_maml))
mus2 = OrderedDict([('model.features.conv1', 0.11651492863893509),
             ('model.features.conv2', 0.27015742659568787),
             ('model.features.conv3', 0.491171270608902),
             ('model.features.conv4', 0.5260140299797058),
             ('model.cls', 0.27396562695503235)])
stds2 = OrderedDict([('model.features.conv1', 0.0060066902078688145),
             ('model.features.conv2', 0.009168228134512901),
             ('model.features.conv3', 0.014589996077120304),
             ('model.features.conv4', 0.023802923038601875),
             ('model.cls', 0.06232181563973427)])
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

x = list(range(1, len(mus2)+1))
x_vals_as_symbols = [layer_name.split('.')[-1].capitalize() for layer_name in mus2.keys()]
x_vals_as_symbols[-1] = 'Head'
title: str = f'Representation distance with metric {metric.upper()}'

if feature_layers_only:
    x = x[:-1]
    x_vals_as_symbols = x_vals_as_symbols[:-1]
    # mus1 = list(mus1.values())[:-1]
    # stds1 = list(stds1.values())[:-1]
    mus2 = list(mus2.values())[:-1]
    stds2 = list(stds2.values())[:-1]
    mus3 = list(mus3.values())[:-1]
    stds3 = list(stds3.values())[:-1]
    title: str = f'Representation distance with metric {metric.upper()} (feature layer only)'
else:
    # mus1 = list(mus1.values())
    # stds1 = list(stds1.values())
    mus2 = list(mus2.values())
    stds2 = list(stds2.values())
    mus3 = list(mus3.values())
    stds3 = list(stds3.values())

# - d(f_sl, A(f_maml))
# plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus1)),
#                       yerr=np.asarray(list(stds1)),
#                       xlabel='Layer', ylabel=f'Distance {metric.upper()}',
#                       title=title, curve_label='SL vs Adapted MAML', ylim=ylim)
# - d(LR(f_sl), A(f_maml))
plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus2)),
                      yerr=np.asarray(list(stds2)),
                      xlabel='Layer', ylabel=f'Distance {metric.upper()}',
                      title=title, curve_label='SL Adapted Head vs Adapted MAML', ylim=ylim)
# - d(f_maml, A(f_maml))
# plot_with_error_bands(x_vals_as_symbols=x_vals_as_symbols, x=x, y=np.asarray(list(mus3)),
#                       yerr=np.asarray(list(stds3)),
#                       xlabel='Layer', ylabel=f'Distance {metric.upper()}',
#                       title=title, curve_label='MAML vs Adapted MAML', ylim=ylim)


save_to_desktop(plot_name=f'lr_sl_vs_maml_maml_{metric}_{feature_layers_only=}')
plt.show()
# plt.clf()
# plt.close()