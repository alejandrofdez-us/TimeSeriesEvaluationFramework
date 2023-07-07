class MetricConfig:
    # FIXME: eliminar window_selection_metric, stride
    def __init__(self, metrics=None, stride=1,
                 window_selection_metric='dtw'):
        if metrics is None:
            metrics = ['js', 'mmd', 'kl', 'ks', 'dtw', 'cc', 'cp', 'hi']
        self.stride = stride
        self.window_selection_metric = window_selection_metric
        self.metrics = metrics
