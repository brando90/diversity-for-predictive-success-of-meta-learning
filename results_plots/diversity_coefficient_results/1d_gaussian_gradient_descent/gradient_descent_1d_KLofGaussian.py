# Calculating the diversity coefficient (psuedocode)
#def dv(Nt, sigma=(mu_b,sigma_b)):
#for i in range(N) {
#f_i ~ p(f | theta_Bf1)
#f1 = identity, f2 = identity
# mu_tau1, mu_tau2 ~ N(mu_b, sigma_b)
# sigma_tau1, sigma_tau2 ~ N(mu_b, sigma_b)
# X1 = array, X2 = array
# for i in 1..M:
#     X1[i] ~ N(mu_tau1, sigma_tau1)
#     X2[i] ~ N(mu_tau2, sigma_tau2)
# d_tau1,tau_2 = 1 - torch.pearson(X1, X2)
# ds.append(d_tau1,tau_2 )
# return torch.mean(ds)
# }

# Then run sgd on dv w.r.t sigma to find the best parameters
import torch
from torch.distributions import normal
import wandb
import numpy as np
run = wandb.init(project="Gradient Descent for Diversity of 1d Gaussian (KL Divergence)", entity="patrickyu")

'''
def pearsonr(x,y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

def covariance(x,y,M):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / M # M = number of elemens in vector vx andvy
'''

def KL(mu1,sigma1,mu2,sigma2):
    #returns asymmetric KL divergence between N(mu1, sigma1) and N(mu2,sigma2)
    #that is, return KL(N(mu1, sigma1) || N(mu2,sigma2))
    # see https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians for eqn details
    return torch.log(sigma2 / sigma1) + (sigma1.pow(2) + (mu1-mu2).pow(2))/(2*sigma2.pow(2)) - 0.5

def dv(N, M, mu_b, sigma_b):
    #divs = []
    dv_avg = torch.tensor(0.0, requires_grad=True)
    for i in range(N):
        mu_tau1 = mu_b + sigma_b * torch.randn(1)#b_dist.sample()#torch.normal(mu_b, sigma_b)
        mu_tau2 = mu_b + sigma_b * torch.randn(1) #b_dist.sample()#torch.normal(mu_b, sigma_b)
        sigma_tau1 = torch.abs((mu_b + sigma_b * torch.randn(1))) #torch.normal(mu_b, sigma_b)
        sigma_tau2 = torch.abs((mu_b + sigma_b * torch.randn(1))) #torch.normal(mu_b, sigma_b)

        #X1 = mu_tau1 + sigma_tau1 * torch.normal(0,1,size=(M,))
        #X2 = mu_tau2 + sigma_tau2 * torch.normal(0,1,size=(M,))
        #print(X1.shape, X2.shape)
        div_coeff = (KL(mu_tau1,sigma_tau1,mu_tau2,sigma_tau2) + KL(mu_tau2,sigma_tau2,mu_tau1,sigma_tau1))/2#1 - pearsonr(X1, X2)
        dv_avg = dv_avg +  div_coeff / N
        #divs.append(div_coeff)

    return dv_avg#, divs

#Run gradient descent, with inital values

N = 1000
M = 1000
mu_0 = 1.0
sigma_0 = 1.0
steps = 100 #number of steps/"epochs"
lr = 0.0001 # learning rate

mu=torch.tensor(mu_0, requires_grad=True)
sigma=torch.tensor(sigma_0, requires_grad=True)
optim = torch.optim.SGD([mu,sigma], lr=lr)

mu_sig = []
mu_sig_grad = []
divs = []

for i in range(steps):
    diversity = dv(N,M,mu,sigma)
    (-diversity).backward() #gradient ascent - we want to increase diversity
    print("new diversity coeff:", diversity)
    print("mu, sigma: ", mu, sigma)
    print("mu, sigma gradient: ",mu.grad, sigma.grad)
    print()

    mu_sig.append([mu.item(),sigma.item()])
    mu_sig_grad.append([mu.grad.item(), sigma.grad.item()])
    divs.append(diversity.item())

    optim.step()
    optim.zero_grad()
    #mu = mu - 0.01*mu.grad
    #sigma = sigma - 0.01*sigma.grad

mu_sig = np.array(mu_sig)
mu_sig_grad = np.array(mu_sig_grad)
divs = np.array(divs)
muvsdiv = wandb.Table(data = [[x,y ] for (x,y) in zip(mu_sig[:,0], divs)],columns=["x","y"])
sigmavsdiv = wandb.Table(data = [[x,y ] for (x,y) in zip(mu_sig[:,1], divs)],columns=["x","y"])
run.log({"N": N, "M":M, "mu_0":mu_0,"sigma_0":sigma_0, "epochs":steps, "learning rate":lr, "plot_mu_sigma" : wandb.plot.line_series(
                       xs=range(steps),
                       ys=[mu_sig[:,0],mu_sig[:,1]],
                       keys=["mu", "sigma"],
                       title="mu_B and sigma_B",
                       xname="epoch"),
                        "plot_mu_sigma_grad" : wandb.plot.line_series(
                                              xs=range(steps),
                                              ys=[mu_sig_grad[:,0],mu_sig_grad[:,1]],
                                              keys=["mu_grad","sigma_grad"],
                                              title="gradient of mu_B and sigma_B",
                                              xname="epoch"),
                         "plot_diversity" : wandb.plot.line_series(
                                               xs=range(steps),
                                               ys=[divs],
                                               keys=["div coeff"],
                                               title="Diversity coefficient",
                                               xname="epoch"),
                        "mu_vs_diversity" : wandb.plot.scatter(muvsdiv, "x", "y", title="mu vs diversity"),
                        "sigma_vs_diversity" : wandb.plot.scatter(sigmavsdiv, "x", "y", title="sigma vs diversity")
                        })
