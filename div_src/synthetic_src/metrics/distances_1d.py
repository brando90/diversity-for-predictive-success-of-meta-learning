"""



"""
from torch import Tensor


def distance_hellinger(p, q, x_lb: int = -1, x_ub: int = 1) -> Tensor:
    """

    H(p, q) = sqrt( 1/2 int_{x \in X} (sqrt(p(x)) - sqrt(q(x))^2 )

    note:
        - trapezoid area: (base1 + base2)/2 * height
    """
    pass


# - tests

def hellinger1d_test():
    pass


if __name__ == '__main__':
    hellinger1d_test()
    print('Done!\a')
