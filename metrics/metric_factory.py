import inspect
import importlib
import os
import json

from metrics.metric import Metric

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MetricFactory(metaclass=Singleton):
    def __init__(self, metrics_to_be_computed, ts2_dict_windowed):
        self.metrics_to_be_computed = metrics_to_be_computed
        self.ts2_dict_windowed = ts2_dict_windowed

        folder_path = os.path.dirname(os.path.abspath(__file__))
        curr_modules = self.__find_classes_in_folder(folder_path)
        self.metric_classes = self.__get_metric_classes(curr_modules)

    def get_metrics_json(self):
        computed_metrics = {}
        for filename, ts_dict in self.ts2_dict_windowed.items():
            computed_metrics[filename] = {}
            for metric_to_be_computed in self.metrics_to_be_computed:
                metric = self.__create_metric(metric_to_be_computed)    
                computed_metrics[filename][metric_to_be_computed] = metric.compute(
                    ts_dict["ts1"], ts_dict["ts2"], ts_dict["cached_metric"]
                )

        computed_metrics = json.dumps(computed_metrics, indent=4)

        return computed_metrics


    def __create_metric(self, metric_name):

        if metric_name in self.metric_classes.keys():
            metric_class = self.metric_classes[metric_name]
            return metric_class()
        else:
            raise ValueError('Invalid metric name')

    @staticmethod
    def __get_metric_classes(modules):
        metric_classes = {}

        for name, obj in modules.items():
            if inspect.ismodule(obj):
                if hasattr(obj, name.capitalize()):
                    instance = getattr(obj, name.capitalize())
                    if issubclass(instance, Metric):
                        metric_classes[name] = instance
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
