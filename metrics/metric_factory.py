import inspect
from metrics.metric import Metric
import importlib
import os

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MetricFactory(metaclass=Singleton):
    def __init__(self, metrics_to_be_computed, ts1, ts2):
        self.metrics_to_be_computed = metrics_to_be_computed
        self.ts1 = ts1
        self.ts2 = ts2

        folder_path = os.path.dirname(os.path.abspath(__file__))
        curr_modules = self.__find_classes_in_folder(folder_path)
        self.metric_classes = self.__get_metric_classes(curr_modules)

    def get_metrics_json(self):
        computed_metrics = {}
        for metric_to_be_computed in self.metrics_to_be_computed:
            metric = self.__create_metric(metric_to_be_computed)    
            computed_metrics[metric_to_be_computed] = metric.compute(
                self.ts1, self.ts2
            )

        return computed_metrics


    def __create_metric(self, metric_name):
        metric_name_lower = metric_name.lower()

        if metric_name_lower in self.metric_classes.keys():
            metric_class = self.metric_classes[metric_name_lower]
            return metric_class()
        else:
            raise ValueError('Invalid metric name')

    @staticmethod
    def __get_metric_classes(modules):
        metric_classes = {}

        for name, obj in modules.items():
            if inspect.ismodule(obj):
                if hasattr(obj, name.upper()):
                    instance = getattr(obj, name.upper())
                    if issubclass(instance, Metric):
                        metric_classes[name.lower()] = instance
        return metric_classes
    
    def __find_classes_in_folder(self, folder_path):
        classes = {}

        for _, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.py'):
                    module_name = file_name[:-3]
                    if module_name in self.metrics_to_be_computed:
                        try:
                            module = importlib.import_module(f".{module_name}", package="metrics")
                            classes[module_name] = module
                        except (ImportError, AttributeError):
                            pass

        return classes
