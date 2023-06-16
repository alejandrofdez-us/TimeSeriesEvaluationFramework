from metrics.metric import Metric
import numpy as np
import sys

class JS(Metric):
    def compute(self, ts1, ts2):
        metric_result = f"Multivariate: {self.__js_distance_multivariate(ts1,ts2)}"

        for column in range(ts2.shape[1]):
            metric_result = metric_result + f" Column {column}: {self.__js_distance(ts1[:, column].reshape(-1, 1), ts2[:, column].reshape(-1, 1))}"

        return metric_result

    def __js_distance(self, ts1, ts2, num_bins=100):
        KL_p_m, KL_q_m = self.__kl_divergence_univariate(ts1, ts2, num_bins=num_bins)
        JS_p_q = (KL_p_m + KL_q_m) / 2
        return JS_p_q


    def __js_distance_multivariate(self, ts1, ts2):
        kl_diverenge_1 = self.__kl_divergence(ts1, ts2)
        kl_diverenge_2 = self.__kl_divergence(ts2, ts1)
        return (kl_diverenge_1 + kl_diverenge_2) / 2
    
    def __kl_divergence_univariate(self, array_1, array_2, range_values=None, num_bins=10):
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


    def __kl_divergence(self, x, y):
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
