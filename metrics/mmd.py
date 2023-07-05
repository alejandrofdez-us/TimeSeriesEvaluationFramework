from sklearn import metrics

from metrics.metric import Metric

class Mmd(Metric):

    def compute(self, ts1, ts2, cached_metric):
        if cached_metric[0] == "mmd":
            metric_result = {"Multivariate": cached_metric[1]}

        else:
            metric_result = {"Multivariate": self.__mmd_calculate_rbf(ts1, ts2)}

        for column in range(ts2.shape[1]):
            metric_result.update({f"Column {column}": self.__mmd_calculate_rbf(ts1[:, column].reshape(-1, 1), ts2[:, column].reshape(-1, 1))})

        return metric_result

    def compute_distance(self, ts1, ts2):
        return self.__mmd_calculate_rbf(ts1, ts2)

    def __mmd_calculate_rbf(self, X, Y, gamma=1.0):
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()
