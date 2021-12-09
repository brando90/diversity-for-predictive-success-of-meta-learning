import torch
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from anatome import *

#import wandb
#run = wandb.init(project="Diversity of 1d Gaussian Other Methods", entity="patrickyu")
# List of all diversity coeffs
l = []

# number of "experiments" in calculating diversity between two Gaussian clusters
B = 5000

# size of each cluster
M = 5000

# spread of each cluster (start w/ sigma = 0.1, then 0.5)
# What if the spread of the two clusters different?
sigma = 0.01

#method to be used
method = "OPD" # SVCCA OPD

for b in range(B):
    mu1 = np.random.standard_normal()
    mu2 = np.random.standard_normal()
    sigma1 = sigma
    sigma2 = sigma

    M1 = torch.normal(mu1, sigma1, size=(M,1))
    M2 = torch.normal(mu2, sigma2, size=(M,1))
    if(method == "LINCKA"):
        results = similarity.linear_cka_distance(M1, M2, reduce_bias=False) #linear cka correlation instead of pearson
    elif(method == "SVCCA"):
        results = similarity.svcca_distance(M1,M2, accept_rate=0.99, backend='svd')
    elif(method == "OPD"):
        results = similarity.orthogonal_procrustes_distance(M1,M2)
    elif(method == "PWCCA"):
        results = similarity.pwcca_distance_choose_best_layer_matrix(M1,M2, backend='svd', epsilon=1e-10)
    print(results)
    diversity_coeff = 1-results
    l += [diversity_coeff] # Append  diversity coeff

l = np.array(l)
print(" B: ", B, " M: ", M, " sigma: ", sigma)
print("All diversity coeff: ", l)
print("Diversity coeff Mean: ", np.mean(l))
print("Diversity coeff Std: ", np.std(l))
print("Method: ", method)
#run.log({"B" : B, "M" : M, "sigma" : sigma, "All diversity coeff": l, "Diversity coeff mean" : np.mean(l), "Diversity coeff std" : np.std(l),"Method: " : method})

#NOTE: make sure l's mean converges, with a small std. or else rerun the experiment!

# TODO: plot sigma vs diversity coeff
# TODO: plot sigma1, sigma2 vs diversity coeff (extension?)
#run.finish()
