from SimilarityAnalysisComputer import SimilarityAnalysisComputer


class MetricComputer(SimilarityAnalysisComputer):

    def compute_next_analysis(self):
        metric = next(self.analysis_iterator)
        filename, ts_dict = self.current_associated_window
        if metric.get_name() in ts_dict["cached_metric"].keys():
            computed_metric = ts_dict["cached_metric"][metric.get_name()]
        else:
            computed_metric = metric.compute(ts_dict["most_similar_ts1_sample"], ts_dict["ts2"])
        return filename, metric.get_name(), computed_metric
