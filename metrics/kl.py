# https://mail.python.org/pipermail/scipy-user/2011-May/029521.html
import sys

import numpy as np
import math


def KLDivergenceUnivariate(array_1, array_2, range_values=None, num_bins=10):
    # smoothing implementes as per https://www.cs.bgu.ac.il/~elhadad/nlp09/KL.html
    # pc = eps*|SU-SP|/|SP| and qc = eps*|SU-SQ|/|SQ|.
    # eps=0.0001
    # SP and SQ the samples observed in P and Q respectively
    # SU = SP U SQ
    # pc = eps*|SU-SP|/|SP| and qc = eps*|SU-SQ|/|SQ|.
    # P'(i) = P(i) - pc if i in SP
    # P'(i) = eps otherwise for i in SU - SP
    eps = 0.000001
    min_array1 = array_1.min()
    min_array2 = array_2.min()
    min_all = min(min_array1, min_array2)
    max_array1 = array_1.max()
    max_array2 = array_2.max()
    max_all = max(max_array1, max_array2)
    range_values = range_values if range_values is not None else (min_all, max_all)
    p = np.histogram(array_1, bins=np.linspace(range_values[0], range_values[1], num_bins + 1))[0] / len(array_1)
    q = np.histogram(array_2, bins=np.linspace(range_values[0], range_values[1], num_bins + 1))[0] / len(array_2)
    pc = eps * (num_bins - (p != 0).sum()) / (p != 0).sum()
    pq = eps * (num_bins - (q != 0).sum()) / (q != 0).sum()
    p = np.vectorize(lambda p_i: eps if p_i == 0 else p_i - pc)(p)
    q = np.vectorize(lambda q_i: eps if q_i == 0 else q_i - pq)(q)
    KL_p_m = sum([p[i] * np.log(p[i] / q[i]) for i in range(len(p))])
    KL_q_m = sum([q[i] * np.log(q[i] / p[i]) for i in range(len(p))])
    return KL_p_m, KL_q_m


def JSdistance(array_1, array_2, num_bins=100):
    KL_p_m, KL_q_m = KLDivergenceUnivariate(array_1, array_2, num_bins=num_bins)
    JS_p_q = (KL_p_m + KL_q_m) / 2
    return JS_p_q


def JSdistanceMultivariate(array_1, array_2):
    kl_diverenge_1 = KLdivergence(array_1, array_2)
    kl_diverenge_2 = KLdivergence(array_2, array_1)
    return (kl_diverenge_1 + kl_diverenge_2) / 2


def KLdivergence(x, y):
    """Compute the Kullback-Leibler divergence between two multivariate samples.

  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.

  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).

  References
  ----------
  P??rez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
    from scipy.spatial import cKDTree as KDTree

    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n, d = x.shape
    m, dy = y.shape

    assert (d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)
    eps = 0.000001

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:, 1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]
    np.set_printoptions(threshold=sys.maxsize)

    pr = eps * (len(r) - (r != 0).sum()) / (r != 0).sum()
    ps = eps * (len(s) - (s != 0).sum()) / (s != 0).sum()
    r = np.vectorize(lambda r_i: eps if r_i == 0 else r_i - pr)(r)
    s = np.vectorize(lambda s_i: eps if s_i == 0 else s_i - ps)(s)

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r / s).sum() * d / n + np.log(m / (n - 1.))
