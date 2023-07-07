class MetricConfig:
    def __init__(self, metrics_names_to_be_computed=None):
        if metrics_names_to_be_computed is None:
            metrics_names_to_be_computed = ['js', 'mmd', 'kl', 'ks', 'dtw', 'cc', 'cp', 'hi']
        self.metrics = metrics_names_to_be_computed
