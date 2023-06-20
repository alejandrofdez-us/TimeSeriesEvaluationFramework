# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.

from sklearn import metrics

from metrics.metric import Metric


class Mmd(Metric):

    def compute(self, ts1, ts2, computed_chosen_metric):
        if computed_chosen_metric[0] == "mmd":
            metric_result = {"Multivariate": computed_chosen_metric[1]}

        else:
            metric_result = {"Multivariate": self.__mmd_calculate_rbf(ts1, ts2)}

        for column in range(ts2.shape[1]):
            metric_result.update({f"Column {column}": self.__mmd_calculate_rbf(ts1[:, column].reshape(-1, 1), ts2[:, column].reshape(-1, 1))})
        
        return metric_result
    
    def compute_distance(self, ts1, ts2):
        return self.__mmd_calculate_rbf(ts1, ts2)
    

    def __mmd_calculate_rbf(self, X, Y, gamma=1.0):
        """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]

        Keyword Arguments:
            gamma {float} -- [kernel parameter] (default: {1.0})

        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.rbf_kernel(X, X, gamma)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
        return XX.mean() + YY.mean() - 2 * XY.mean()

