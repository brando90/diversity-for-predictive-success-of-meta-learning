import torch
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from uutils.torch_uu.metrics.cca import cca_core
import wandb
run = wandb.init(project="Diversity of 1d Gaussian SVCCA", entity="patrickyu")
# List of all diversity coeffs
l = []

# number of "experiments" in calculating diversity between two Gaussian clusters
B = 1000

# size of each cluster
M = 1000

# spread of each cluster (start w/ sigma = 0.1, then 0.5)
# What if the spread of the two clusters different?
sigma = 0.5

for b in range(B):
    mu1 = np.random.standard_normal()
    mu2 = np.random.standard_normal()
    sigma1 = sigma
    sigma2 = sigma

    M1 = np.random.normal(mu1, sigma1, size=(1,M))
    M2 = np.random.normal(mu2, sigma2, size=(1,M))
    results = cca_core.get_cca_similarity(M1, M2) #SVCCA correlation instead of pearson
    r = np.mean(results["cca_coef1"])
    diversity_coeff = 1-r
    l += [diversity_coeff] # Append  diversity coeff

l = np.array(l)
print(" B: ", B, " M: ", M, " sigma: ", sigma)
print("All diversity coeff: ", l)
print("Diversity coeff Mean: ", np.mean(l))
print("Diversity coeff Std: ", np.std(l))
run.log({"B" : B, "M" : M, "sigma" : sigma, "All diversity coeff": l, "Diversity coeff mean" : np.mean(l), "Diversity coeff std" : np.std(l)})

#NOTE: make sure l's mean converges, with a small std. or else rerun the experiment!

# TODO: plot sigma vs diversity coeff
# TODO: plot sigma1, sigma2 vs diversity coeff (extension?)
run.finish()
