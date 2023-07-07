class PlotConfig:
    # FIXME: eliminar window_selection_metric, stride
    def __init__(self, figures=None, timestamp_frequency_seconds=1, stride=1,
                 window_selection_metric='dtw'):
        if figures is None:
            figures = ['2d', 'deltas', 'dtw', 'pca', 'tsne']
        self.stride = stride
        self.window_selection_metric = window_selection_metric
        self.figures = figures
        self.timestamp_frequency_seconds = timestamp_frequency_seconds
