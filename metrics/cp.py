import numpy as np

from metrics.metric import Metric

class Cp(Metric):
    def compute(self, ts1, ts2, cached_metric):
        if cached_metric[0] == "cp":
            metric_result = {"Multivariate": cached_metric[1]}

        else:
            metric_result = {"Multivariate": self.__cp(ts1, ts2)}

        return metric_result
    
    def compute_distance(self, ts1, ts2):
        return self.__cp(ts1, ts2)

    def __cp(self, ts1, ts2):
        # TODO: remove the following two commented lines?
        # normalized_ts1 = normalize_start_time_to_zero(ts1)
        # normalized_ts2 = normalize_start_time_to_zero(ts2)
        ts1_pearson = np.corrcoef(ts1, rowvar=False)
        ts2_pearson = np.corrcoef(ts2, rowvar=False)
        correlation_diff_matrix = ts1_pearson - ts2_pearson
        l1_norms_avg = np.mean([np.linalg.norm(row) for row in correlation_diff_matrix])
        return l1_norms_avg
