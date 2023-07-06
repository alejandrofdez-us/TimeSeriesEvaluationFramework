import os

from metrics.metric import Metric
from reader import find_available_classes

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MetricFactory(metaclass=Singleton):
    def __init__(self, metrics_to_be_computed):
        self.metrics_to_be_computed = metrics_to_be_computed

        self.folder_path = os.path.dirname(os.path.abspath(__file__))
        self.metric_objects = self.find_metrics_in_directory(metrics_to_be_computed, self.folder_path)

    @staticmethod
    def find_metrics_in_directory(metrics_to_be_computed, folder_path):
        available_metrics = MetricFactory.find_available_metrics(folder_path)
        available_metrics = {k: v for k, v in available_metrics.items() if k in metrics_to_be_computed}

        return available_metrics

    @staticmethod
    def find_available_metrics(folder_path):
        available_metrics = find_available_classes(folder_path, Metric, "metrics")
        return available_metrics
