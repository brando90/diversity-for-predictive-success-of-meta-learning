"""


"""
import logging
from math import ceil


def get_num_examples_needed_to_reach_a_certain_saftey_margin(safety_margin: float,
                                                             HW: int,  # e.g. HW or 1 for cls
                                                             effective_dims_D: int,  # e.g. C=num_filters or num_classes
                                                             ) -> int:
    """
    Return the smallest number of examples needed to have a none-pathological similarity metric.

    For CNN layer use:
        HW = H*W (assumes each path is an image resulting in a data matrix of size [M*H*W,C])
    For cls layer use:
        HW = 1 (since the shape of the cls layer is [B, num_classes]

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

# %%
# MI, Resnet12

batch_size = 25
n_ways = 5
HW = 640


print()

# %%
# %%
# Cifar-fs, Resnet12
HW = 4096
