class MetricConfig:
    def __init__(self, metrics=None):
        if metrics is None:
            metrics = ['js', 'mmd', 'kl', 'ks', 'dtw', 'cc', 'cp', 'hi']
        self.metrics = metrics
