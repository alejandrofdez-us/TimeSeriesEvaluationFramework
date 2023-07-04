class PlotConfig:
    def __init__(self, stride, window_selection_metric, figures, timestamp_frequency_seconds):
        self.stride = stride
        self.window_selection_metric = window_selection_metric
        self.figures = figures
        self.timestamp_frequency_seconds = timestamp_frequency_seconds
