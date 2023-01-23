"""
File for estimating the sample size needed to get a certain P_d (probability of detecting a difference, Power)
for a t-test using the p-value decision procedure.

Note:
    P_D = 1 - P_M (probability of making a mistake, Type II error) - Pr[H(y) = H1 | H = H1] = Pr[we decide H1/reject null under the H0 null hypothesis model].
    beta = P_M = Pr[H0 | H1] = 1 - P_D
    p-value = how likely is the data we see or more extreme under the null hypothesis model i.e. Pr[H1 | H0] so it's the false positive rate/type 1/false alarm P_f.

ref:
    - https://en.wikipedia.org/wiki/Power_(statistics)
    - https://stats.oarc.ucla.edu/other/gpower/power-analysis-for-one-sample-t-test/
"""
#%%
"""
Goal: Estimate N given a desired power. Instead getting N, we will plug in N's until we get desired/estimated Power (apriori).
Afaik, we need the std when generating dummy data. From my task2vec hist, we can get it + they look normal already. 
Thus:
  - Given N, std -> Power 
    - tails
    - alpha (significance level)
    - effect size d (difference between means)
    - Power (probability of detecting a difference, 1 - beta)

- ref: 
    - https://stats.oarc.ucla.edu/other/gpower/power-analysis-for-one-sample-t-test/
"""

