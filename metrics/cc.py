import numpy as np

from metrics.metric import Metric

class Cc(Metric):
    def compute(self, ts1, ts2, computed_chosen_metric):
        if computed_chosen_metric[0] == "cc":
            metric_result = {"Multivariate": computed_chosen_metric[1]}
        
        else:
            metric_result = {"Multivariate": self.__cc(ts1, ts2)}

        return metric_result
    
    def compute_distance(self, ts1, ts2):
        return self.__cc(ts1, ts2)

    def __cc(self, ts1, ts2):
        # TODO: remove the following two commented lines?
        # normalized_ts1 = normalize_start_time_to_zero(ts1)
        # normalized_ts2 = normalize_start_time_to_zero(ts2)
        ts1_covariance = np.cov(ts1)
        generated_data_covariance = np.cov(ts2)
        covariance_diff_matrix = ts1_covariance - generated_data_covariance
        l1_norms_avg = np.mean([np.linalg.norm(row) for row in covariance_diff_matrix])
        return l1_norms_avg
