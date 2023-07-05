class Metric:
    def __init__(self):
        self.name = self.__class__.__name__.lower()

    def compute (self, ts1, ts2, cached_metric):
        raise NotImplementedError('Subclasses must implement compute() method')

    def compare(self, metric1, metric2):
        return metric2 - metric1
    