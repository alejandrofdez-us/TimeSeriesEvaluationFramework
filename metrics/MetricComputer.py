class MetricComputer:

    def __init__(self, ts1_ts2_associated_windows, metrics, cached_metric=None):
        self.ts1_ts2_associated_windows_iterator = iter(ts1_ts2_associated_windows.items())
        self.metrics = metrics
        self.metrics_iterator = iter(metrics)
        self.cached_metric = cached_metric
        self.length = len(metrics) * len(ts1_ts2_associated_windows.items())

    def __iter__(self):
        self.current_associated_window = next(self.ts1_ts2_associated_windows_iterator)
        return self

    def __next__(self):
        try:
            return self.__compute_next_metric()
        except StopIteration:
            self.current_associated_window = next(self.ts1_ts2_associated_windows_iterator)
            self.metrics_iterator = iter(self.metrics)
            return self.__compute_next_metric()

    def __len__(self):
        return self.length

    def __compute_next_metric(self):
        metric = next(self.metrics_iterator)
        filename, ts_dict = self.current_associated_window
        if metric.get_name() not in ts_dict["cached_metric"].keys():
            computed_metric = metric.compute(ts_dict["most_similar_ts1_sample"], ts_dict["ts2"])
        else:
            computed_metric = ts_dict["cached_metric"][metric.get_name()]
        return filename, metric.get_name(), computed_metric
