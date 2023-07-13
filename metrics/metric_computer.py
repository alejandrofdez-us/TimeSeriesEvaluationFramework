import warnings
from similarity_analysis_computer import SimilarityAnalysisComputer


class MetricComputer(SimilarityAnalysisComputer):

    def _compute_next_analysis(self):
        metric = next(self.analysis_iterator)
        filename, ts_dict = self.current_associated_window
        if metric.get_name() in ts_dict['cached_metric'].keys():
            computed_metric = ts_dict['cached_metric'][metric.get_name()]
        else:
            try:
                computed_metric = metric.compute(ts_dict['most_similar_ts1_sample'], ts_dict['ts2'])
            except Exception as e:
                computed_metric = []
                warnings.warn(f'\nWarning: Metric {metric.get_name()} could not be computed. Details: {e}', Warning)
        return filename, metric.get_name(), computed_metric
