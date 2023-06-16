import numpy as np

from metrics.metric import Metric

class CP(Metric):
    def compute(self, ts1, ts2):
        self.ts1 = ts1
        self.ts2 = ts2
    
        metric_result = f"Multivariate: {self.__cp()}"

        return metric_result

    def __cp(self):
        # TODO: remove the following two commented lines?
        # normalized_ts1 = normalize_start_time_to_zero(ts1)
        # normalized_ts2 = normalize_start_time_to_zero(ts2)
        ts1_pearson = np.corrcoef(self.ts1, rowvar=False)
        ts2_pearson = np.corrcoef(self.ts2, rowvar=False)
        correlation_diff_matrix = ts1_pearson - ts2_pearson
        l1_norms_avg = np.mean([np.linalg.norm(row) for row in correlation_diff_matrix])
        return l1_norms_avg
