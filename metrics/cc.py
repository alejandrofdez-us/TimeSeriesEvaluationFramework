import numpy as np

from metrics.metric import Metric

class Cc(Metric):
    def compute(self, ts1, ts2):
        self.ts1 = ts1
        self.ts2 = ts2

        metric_result = {"Multivariate": self.__cc()}

        return metric_result

    def __cc(self):
        # TODO: remove the following two commented lines?
        # normalized_ts1 = normalize_start_time_to_zero(ts1)
        # normalized_ts2 = normalize_start_time_to_zero(ts2)
        ts1_covariance = np.cov(self.ts1)
        generated_data_covariance = np.cov(self.ts2)
        covariance_diff_matrix = ts1_covariance - generated_data_covariance
        l1_norms_avg = np.mean([np.linalg.norm(row) for row in covariance_diff_matrix])
        return l1_norms_avg
