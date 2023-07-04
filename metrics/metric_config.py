class MetricConfig:
    def __init__(self, stride, window_selection_metric, metrics):
        self.stride = stride
        self.window_selection_metric = window_selection_metric
        self.metrics = metrics
