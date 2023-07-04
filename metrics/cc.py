import numpy as np

from metrics.metric import Metric

class Cc(Metric):
    def compute(self, ts1, ts2, cached_metric):
        if cached_metric[0] == "cc":
            metric_result = {"Multivariate": cached_metric[1]}
        
        else:
            metric_result = {"Multivariate": self.__cc(ts1, ts2)}

        return metric_result
    
    def compute_distance(self, ts1, ts2):
        return self.__cc(ts1, ts2)

    def __cc(self, ts1, ts2):
        ts1_covariance = np.cov(ts1)
        generated_data_covariance = np.cov(ts2)
        covariance_diff_matrix = ts1_covariance - generated_data_covariance
        l1_norms_avg = np.mean([np.linalg.norm(row) for row in covariance_diff_matrix])
        return l1_norms_avg
