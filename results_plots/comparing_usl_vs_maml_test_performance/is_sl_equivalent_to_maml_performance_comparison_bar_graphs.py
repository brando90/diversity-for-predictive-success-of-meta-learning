# %%
"""

ref:
    - runs (first round of results):
        https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/reports/SL-vs-MAML-MI-Cifarfs-5CNN-Resnet12-preliminary-experiments---VmlldzoxNTcyNDk3
    - runs (2nd round): TODO
"""

# %%
"""
1
MI, 5CNN


details:
- using original (old code). Thus:
- 5CNN is SL rfs 
- 5CNN MAML is adam until convergence my_torchmeta_maml_code 
"""
from uutils.plot import bar_graph_with_error_using_pandas, save_to_desktop
from matplotlib import pyplot as plt

groups = ['MI, 5CNN']  # the rows of a df
adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
meta_test_acc = [62.4, 62.3, 60.1]
meta_test_ci = [1.64, 1.5, 1.37]
# meta_test_acc = [62.4, 62.3, 60.1]
# meta_test_ci = [1.64, 1.5, 1.37]
row1 = meta_test_acc + meta_test_ci
data = [row1]

bar_graph_with_error_using_pandas(group_row_names=groups,
                                  columns=adapted_models,
                                  rows=data,
                                  val_names=adapted_models[0:3],
                                  error_bar_names=adapted_models[3:],
                                  title='Performance Comparsion MAML vs TL (USL)',
                                  xlabel='Dataset, Architecture',
                                  ylabel='Meta-Test Accuracy'
                                  )
save_to_desktop(plot_name='maml_vs_tl_mi_5cnn_perf_comp_bar')
plt.show()

# %%
"""
2
MI, resnet12


details:
- resnet12 is SL rfs with adam my_l2l_code
- resnet12 MAML is adam until convergence my_l2l_maml_code (REDO, use new l2l ckpt)
"""
from uutils.plot import bar_graph_with_error_using_pandas, save_to_desktop
from matplotlib import pyplot as plt

groups = ['MI, ResNet12']  # the rows of a df
adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
meta_test_acc = [73.8, 72.8, 70.8]
meta_test_ci = [1.76, 1.61, 1.70]
row1 = meta_test_acc + meta_test_ci
data = [row1]

bar_graph_with_error_using_pandas(group_row_names=groups,
                                  columns=adapted_models,
                                  rows=data,
                                  val_names=adapted_models[0:3],
                                  error_bar_names=adapted_models[3:],
                                  title='Performance Comparsion MAML vs TL (USL)',
                                  xlabel='Dataset, Architecture',
                                  ylabel='Meta-Test Accuracy'
                                  )
save_to_desktop(plot_name='maml_vs_tl_mi_resnet12rfs_perf_comp_bar')
plt.show()

# %%
"""
3
Cifar-fs, 5CNN1024


details:
- train to convergence & train-acc==0: https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3sxs4q08/logs?workspace=user-brando
"""
from uutils.plot import bar_graph_with_error_using_pandas, save_to_desktop
from matplotlib import pyplot as plt

groups = ['Cifar-fs, 5CNN']  # the rows of a df
adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
meta_test_acc = [76.9, 75.9, 70.1]
meta_test_ci = [1.97, 1.75, 1.48]
row1 = meta_test_acc + meta_test_ci
data = [row1]

bar_graph_with_error_using_pandas(group_row_names=groups,
                                  columns=adapted_models,
                                  rows=data,
                                  val_names=adapted_models[0:3],
                                  error_bar_names=adapted_models[3:],
                                  title='Performance Comparsion MAML vs TL (USL)',
                                  xlabel='Dataset, Architecture',
                                  ylabel='Meta-Test Accuracy'
                                  )
save_to_desktop(plot_name='maml_vs_tl_cifarfs_5cnn_perf_comp_bar')
plt.show()

#%%
"""
4
Cifar-fs, resnet12


details:
- resnet12 is SL rfs with adam my_l2l_code
- resnet12 MAML is adam until convergence my_l2l_maml_code
"""
from uutils.plot import bar_graph_with_error_using_pandas, save_to_desktop
from matplotlib import pyplot as plt

groups = ['Cifar-fs, ResNet12']  # the rows of a df
adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
meta_test_acc = [78.2, 77.4, 75.4]
meta_test_ci = [1.73, 1.78, 1.69]
row1 = meta_test_acc + meta_test_ci
data = [row1]

bar_graph_with_error_using_pandas(group_row_names=groups,
                                  columns=adapted_models,
                                  rows=data,
                                  val_names=adapted_models[0:3],
                                  error_bar_names=adapted_models[3:],
                                  title='Performance Comparsion MAML vs TL (USL)',
                                  xlabel='Dataset, Architecture',
                                  ylabel='Meta-Test Accuracy'
                                  )
save_to_desktop(plot_name='maml_vs_tl_mi_resnet12rfs_perf_comp_bar')
plt.show()

# %%
"""
alla data sets, all archs

wanbds:
- mi resnet12
- mi 5cnn
- cifarfs resnet12
- cifarfs 5cnn1024: https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/runs/3sxs4q08/logs?workspace=user-brando
"""
from uutils.plot import bar_graph_with_error_using_pandas, save_to_desktop
from matplotlib import pyplot as plt

groups = ['MI, 5CNN', 'MI, ResNet12', 'Cifar-fs, 5CNN', 'Cirfar-fs, ResNet12']  # the rows of a df
adapted_models = ['MAML5', 'MAML10', 'USL', 'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
# MI, 5CNN
meta_test_acc = [62.4, 62.3, 60.1]
meta_test_ci = [1.64, 1.5, 1.37]
row1 = meta_test_acc + meta_test_ci
# - MI, resnet12
# old ckpts
meta_test_acc = [73.8, 72.8, 70.8]
meta_test_ci = [1.76, 1.61, 1.70]
# new matching ckpts
# meta_test_acc = [73.8, 72.8, 70.8]
# meta_test_ci = [1.76, 1.61, 1.70]
row2 = meta_test_acc + meta_test_ci
# - cifar-fs, 5CNN1024
meta_test_acc = [76.9, 75.9, 70.1]
meta_test_ci = [1.97, 1.75, 1.48]
row3 = meta_test_acc + meta_test_ci
# - cifar-fs, resnet12
meta_test_acc = [78.2, 77.4, 75.4]
meta_test_ci = [1.73, 1.78, 1.69]
row4 = meta_test_acc + meta_test_ci
#  - join all rows
data = [row1, row2, row3, row4]

bar_graph_with_error_using_pandas(group_row_names=groups,
                                  columns=adapted_models,
                                  rows=data,
                                  val_names=adapted_models[0:3],
                                  error_bar_names=adapted_models[3:],
                                  title='Performance Comparison MAML vs Transfer Learning (USL)',
                                  xlabel='Dataset, Architecture',
                                  ylabel='Meta-Test Accuracy',
                                  loc='upper left'
                                  )
save_to_desktop(plot_name='all_archs_all_data_sets_maml_vs_tl_mi_cirfarfs_5cnn_resnet12rfs_perf_comp_bar')
plt.show()


#%%
"""
wandb report: https://wandb.ai/brando/entire-diversity-spectrum/reports/HDB1-MIO-Performance-Comparison-MAML-vs-USL--VmlldzoyOTA2NjM2/edit
"""
from uutils.plot import bar_graph_with_error_using_pandas, save_to_desktop
from matplotlib import pyplot as plt
from uutils.torch_uu.metrics.confidence_intervals import confidence_intervals_intersect

groups = []
adapted_models = ['MAML5', 'MAML10', 'USL',
                  'MAML5 ci', 'MAML10 ci', 'USL ci']  # columns of a df
data = []

groups += ['MIO, ResNet12']
adapted_models += []

# - MIO prototype, batch-size=1000, https://wandb.ai/brando/entire-diversity-spectrum/runs/4un3y2yo/logs?workspace=user-brando
# meta_test_acc = [78.9, 76.9, 67.8]
# meta_test_ci = [1.36, 1.39, 1.39]
# row = meta_test_acc + meta_test_ci
# data += [row]

# - MIO, better usl ckpt acc=0.9996, https://wandb.ai/brando/entire-diversity-spectrum/runs/37uvzvrq/logs?workspace=user-brando
meta_test_acc = [78.9, 76.9, 81.4]
meta_test_ci = [1.36, 1.39, 1.09]
row = meta_test_acc + meta_test_ci
data += [row]
pairs = list(zip(meta_test_acc, meta_test_ci))
print(f'{pairs=}')
maml5_vs_usl = confidence_intervals_intersect(pairs[0], pairs[2])
maml10_vs_usl = confidence_intervals_intersect(pairs[1], pairs[2])
print(f'maml5 vs usl do NOT intersect (yes seperation of maml5 vs usl?): {not maml5_vs_usl}')
print(f'maml10 vs usl do NOT intersect (yes seperation of maml5 vs usl?): {not maml10_vs_usl}')
print(f'both don\'t intersect (usl vs maml separation?): {not maml5_vs_usl and not maml10_vs_usl}')

# - MIO, higher 2k num tasks: https://wandb.ai/brando/entire-diversity-spectrum/runs/614hwt54/logs?workspace=user-brando
# meta_test_acc = [82.75, 84.42, 85.25]
# meta_test_ci = [0.7423, 0.7263, 0.6408]
# row = meta_test_acc + meta_test_ci
# data += [row]
# pairs = list(zip(meta_test_acc, meta_test_ci))
# print(f'{pairs=}')
# maml5_vs_usl = confidence_intervals_intersect(pairs[0], pairs[2])
# maml10_vs_usl = confidence_intervals_intersect(pairs[1], pairs[2])
# print(f'maml5 vs usl do NOT intersect (yes seperation of maml5 vs usl?): {not maml5_vs_usl}')
# print(f'maml10 vs usl do NOT intersect (yes seperation of maml5 vs usl?): {not maml10_vs_usl}')
# print(f'both don\'t intersect (usl vs maml separation?): {not maml5_vs_usl and not maml10_vs_usl}')
# USL vs maml not seperated because maml10 & usl intersect

# - MIO, higher 5k num tasks: https://wandb.ai/brando/entire-diversity-spectrum/runs/33bvr8eb/logs?workspace=user-brando ampere1
# meta_test_acc = [83.05, 83.04, 84.59]
# meta_test_ci = [0.4827, 0.4849, 0.41238]
# row = meta_test_acc + meta_test_ci
# data += [row]
# pairs = list(zip(meta_test_acc, meta_test_ci))
# print(f'{pairs=}')
# maml5_vs_usl = confidence_intervals_intersect(pairs[0], pairs[2])
# maml10_vs_usl = confidence_intervals_intersect(pairs[1], pairs[2])
# print(f'maml5 vs usl do NOT intersect (yes seperation of maml5 vs usl?): {not maml5_vs_usl}')
# print(f'maml10 vs usl do NOT intersect (yes seperation of maml5 vs usl?): {not maml10_vs_usl}')
# print(f'both don\'t intersect (usl vs maml separation?): {not maml5_vs_usl and not maml10_vs_usl}')
# just barely don't intersect

# -

bar_graph_with_error_using_pandas(group_row_names=groups,
                                  columns=adapted_models,
                                  rows=data,
                                  val_names=adapted_models[0:3],
                                  error_bar_names=adapted_models[3:],
                                  title='Performance Comparison MAML vs Transfer Learning (USL)',
                                  xlabel='Dataset, Architecture',
                                  ylabel='Meta-Test Accuracy',
                                  loc='upper left'
                                  )
# save_to_desktop(plot_name='mio_hdb1_prototype')
save_to_desktop(plot_name='mio_hdb1_maml_98_usl_99')
# save_to_desktop(plot_name='mio_hdb1_5k')
plt.show()
