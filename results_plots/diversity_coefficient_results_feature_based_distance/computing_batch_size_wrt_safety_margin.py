"""

Rep layer:
want for each dist btw tasks to be valid in [B,n*k,C,H,W] by choosing the k with saftey at least 5,10,20. i.e.
    choose s(k|n,HW,C) by varying k s.t. n*k*HW >= s(k)*C
so:
    s(k|n,HW,C) >= 5, 10, 20
note that the B dim plays no role since it's the index for the distance btw each task d(tau1, tau2).
    d:[M,C,H,W]^2 -> [1]
    d:task^2 -> dist
and we get B of these to compute div by taking mean of them.

Cls:
note:
    - for meta-learning we have [B, M, n_cls] and the meta-batch dim is not part of this calculation
    since the distance is calculate per task (i.e. per [M, n_cls] layer_matrix). So we use M in the
    previous reasoning for B. So we do:
        s(M, n_cls) = M/n_cls
    and we want to choose M s.t. s(M, n_cls) >= 5, 10 or 20.
    M=n*k.

"""
import logging
from math import ceil


def s_cnn(k, n, HW, C):
    """
    Given k, n, HW, C gets the safe computation of task distance for a cnn
    (using patches as images & neurons as dims/features).
    Based on:
        n*k*HW >= s*C
        n*k*HW/ C >= s
        s(k|n, HW, C) >= s
    where s is desired safety margin.

    Note: usually you'd want this number to be greater than 5,10, 20.
    You can play around with k until you get the desired range.
    Note: This code is mostly to conceptually show what is going on. Brute for search is unnecessary.
    Note: [B, M, C,H,W] for meta-learning, the number B plays no role since you get a dist for every
    pair of tasks of size [M, C,H,W] for every b \in [B]. Then you compute the div by taking expectation over
    [B].
    """
    return n*k*HW / C

def s_cls(M, n_cls):
    """
    Given number of example M and number of classification classes gives safe distance computation of cls layer/head.
    Note: This code is mostly to conceptually show what is going on. Brute for search is unnecessary.
    Based on:
        M >= s*n_cls
        M/n_cls >= s
    note: [B, M, n_cls] the B for meta-learning plays no role in computing safety value for distance between tasks.
    note: M = n*k for cls layer.
    note: if meta-learning M=n*k_eval so s = M/n_cls = n*k/n_cls = k_eval >= s.
        So you can use safety margin as number of k_eval for n_cls layer in meta-learning for few-shot leanring.
    """
    return M / n_cls

def get_num_examples_needed_to_reach_a_certain_saftey_margin(safety_margin: float,
                                                             HW: int,  # e.g. HW or 1 for cls
                                                             effective_dims_D: int,  # e.g. C=num_filters or num_classes
                                                             ) -> int:
    """
    Return the smallest number of examples needed to have a none-pathological similarity metric.

    For CNN layer use:
        HW = H*W (assumes each patch is an image resulting in a data matrix of size [M*H*W,C])
    For cls layer use:
        HW = 1 (since the shape of the cls layer is [M, num_classes]

    Number of examples needed for distance computations to be safe i.e. not collapse because there are
    less data points than features/dim.
    Formally for effective # of data points N' and it's corresponding dim/features D':
        N' >= s*D'
    For CNNs do:
        MHW >= s*C
    since each image patch can be considered an image point and thus it's number of neurons/filters the number of dims.
    For cls layer (just standard matrix):
        N' >= s*D'
        M >= s*out_features
    Recommended safety margings:
        s = 5, 10, 20.
    If you use >= 20 you are essentially safe from the collapse. i.g. you need at least N' > D' but if you are a bit
    more than that like N' >= sD' that is the best. If you have N' larger than num features by some amount it means that
    you def have enough data to avoid having too high sim which leads to pathologically low distances which could lead
    to pathologically low divs.


    Method:

    B*k_eval*

    :return:  M - the number of examples needed.
    """
    if safety_margin < 5:
        logging.warning(f'Your saftey marging is low, at least 5 is recommended but its: {safety_margin=}.')
    elif safety_margin < 1:
        raise ValueError(f'Safety margin has to be at least 1 otherwise your distance & div will always be zero.')
    else:
        assert safety_margin >= 5
        D: int = effective_dims_D
        # want MHW >= s*C -> M >= (s*C/HW)
        lower_bound: float = ((safety_margin * D) / HW)
        return ceil(lower_bound)

# %%
# MI, 5CNN

batch_size = 25
n_ways = 5
HW = -1


# %%
# MI, Resnet12
"""
"""

batch_size = 25
n_ways = 5
HW = 640
C = 1  # TODO: get number of filters at the rep layer

# number we want is k_eval in [n*k_eval*H*W, C] s.t. n*k_eval*H*W >= s*D (N' >= s*D')
D_effective = C


print()

# %%
# %%
# Cifar-fs, Resnet12
HW = 4096
