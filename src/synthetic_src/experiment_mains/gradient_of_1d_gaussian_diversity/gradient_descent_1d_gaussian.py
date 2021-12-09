
# Calculating the diversity coefficient
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

def pearsonr(x,y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

def dv(N, M, mu_b, sigma_b):
    #divs = []
    dv_avg = torch.tensor(0.0, requires_grad=True)
    for i in range(N):
        # mu_tau1, mu_tau2 ~ N(mu_b, sigma_b)
        b_dist = normal.Normal(mu_b,sigma_b)
        mu_tau1 = b_dist.sample()#torch.normal(mu_b, sigma_b)
        mu_tau2 = b_dist.sample()#torch.normal(mu_b, sigma_b)
        sigma_tau1 = b_dist.sample()**2 #torch.normal(mu_b, sigma_b)
        sigma_tau2 = b_dist.sample()**2 #torch.normal(mu_b, sigma_b)

        #X1 = torch.normal(mu_tau1,sigma_tau1,size=(M,))
        #X2 = torch.normal(mu_tau2,sigma_tau2,size=(M,))
        #print(mu_tau1, sigma_tau1)
        X1_dist = normal.Normal(mu_tau1,sigma_tau1)
        X2_dist = normal.Normal(mu_tau1,sigma_tau1)
        X1 = X1_dist.sample([M])
        X2 = X2_dist.sample([M])

        div_coeff = 1 - pearsonr(X1, X2)
        dv_avg = dv_avg +  div_coeff / N
        #divs.append(div_coeff)

    return dv_avg#, divs

#Run gradient descent, with inital values
mu=torch.tensor(0.0, requires_grad=True)
sigma=torch.tensor(0.5, requires_grad=True)
div_init = dv(1000,1000,mu,sigma)
print(div_init)
div_init.backward()
print(mu.grad, sigma.grad)
