from __future__ import division
import numpy as np
import numpy.linalg as LA


def proj_ball(x, center, rad):
    """
    Compute projection of x onto the closed ball B(center, rad)
    """
    d = x - center
    dist = LA.norm(d)
    if dist <= rad:
        return x
    else:
        return rad * d / dist + center


def prox_norm_1(x, eps, u=0):
    """
    Find proximal operator of function eps*||x-u||_1
    """
    x1 = x + np.clip(u - x, -eps, eps)
    return x1


def prox_norm_2(x, eps, a=0):
    """
    Find proximal operator of function f = 0.5*eps*||x-a||**2,
    """
    return (x + eps * a) / (eps + 1)


def project_nd(x, r=1):
    '''perform a pixel-wise projection onto r-radius balls. Here r = 1'''
    norm_x = np.sqrt(
        (x * x).sum(-1))  # Calculate L2 norm over the last array dimension
    nx = np.maximum(1.0, norm_x / r)
    return x / nx[..., np.newaxis]


def proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    Implementation by Adrien Gaidon - INRIA - 2011.
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w
