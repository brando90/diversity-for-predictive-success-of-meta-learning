"""
Computing div according to task2vec:
- https://github.com/awslabs/aws-cv-task2vec/blob/master/small_datasets_example.ipynb
- https://wandb.ai/brando/sl_vs_ml_iclr_workshop_paper/reports/Rough-estimate-of-diversity-of-task2vec-union-data-sets--VmlldzoxMjQxNDk4
"""
import numpy as np

dists: list[float] = []
dists += 3*[0.15]
dists += 3*[0.6]
dists += 6*[0.4]
dists += [0.05]
dists += 2*[0.3]
dists += dists

print(f'{dists=}')
print(f'{len(dists)=}')
print(f'div_mu = {np.mean(dists)=}')
print(f'div_std = {np.std(dists)=}')
"""
dists=[0.15, 0.15, 0.15, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.05, 0.3, 0.3, 0.15, 0.15, 0.15, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.05, 0.3, 0.3]
len(dists)=30
div_mu = np.mean(dists)=0.35333333333333344
div_std = np.std(dists)=0.1667999467092907
"""


"""
to implement the distance of tasks according to task2vec do:
- compute the distance matrix (rows, columns are the tasks or data sets) e.g. X1, X2 from MI
- then given that matrix compute the diversity (e.g. remove the diagonal then do it) mu and std
- print div_mu, div_std
"""


#%%
"""
diversity for benchmark = HD4ML_1.

HDML_1 = ('stl10', 'letters', 'kmnist')
"""
import numpy as np

dists: list[float] = []
# stl10 vs letters, kmnist
dists += [0.6, 0.52]
# kmnist vs letters
dists += [0.22]

print('-- OUTPUT --')
print(f'{dists=}')
print(f'{len(dists)=}')
print(f'div_mu = {np.mean(dists)=}')
print(f'div_std = {np.std(dists)=}')