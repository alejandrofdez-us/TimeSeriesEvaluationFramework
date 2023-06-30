import inspect
import importlib
import os

from plots.plot import Plot

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class PlotFactory(metaclass=Singleton):
    def __init__(self, figures_to_be_generated, args):
        self.figures_to_be_generated = figures_to_be_generated
        self.args = args
        self.pca_is_computed = False
        self.tsne_is_computed = False
        self.figures_requires_all_samples = []
        self.computed_figures_requires_all_samples= []

        folder_path = os.path.dirname(os.path.abspath(__file__))
        curr_modules = self.__find_classes_in_directory(folder_path)
        self.figure_classes = self.__get_figure_classes(curr_modules)


    def generate_figures(self):
        generated_plots = {}
        for filename, figure_to_be_computed_args in self.args.items():
            generated_plots[filename] = {}
            for figure_to_be_computed in self.figures_to_be_generated:
                if (figure_to_be_computed not in self.computed_figures_requires_all_samples):
                    plot = self.__create_figure(figure_to_be_computed)
                    generated_plots[filename][figure_to_be_computed] = plot.generate_figures(
                        figure_to_be_computed_args
                    )
                    if figure_to_be_computed in self.figures_requires_all_samples:
                        self.computed_figures_requires_all_samples.append(figure_to_be_computed)

        return generated_plots


    def __create_figure(self, figure_to_be_computed):

        if figure_to_be_computed in self.figure_classes.keys():
            figure_class = self.figure_classes[figure_to_be_computed]
            return figure_class()
        else:
            raise ValueError('Invalid metric name')

    def __get_figure_classes(self, modules):
        figure_classes = {}

        for name, obj in modules.items():
            if inspect.ismodule(obj):
                if hasattr(obj, name.capitalize()):
                    instance = getattr(obj, name.capitalize())
                    if issubclass(instance, Plot):
                        figure_classes[name] = instance
                        if instance.requires_all_samples():
                            self.figures_requires_all_samples.append(name)
        return figure_classes
    
    def __find_classes_in_directory(self, folder_path):
        classes = {}

        for _, _, files in os.walk(folder_path):
            for file_name in files:
                if file_name.endswith('.py'):
                    module_name = file_name[:-3]
                    if module_name in self.figures_to_be_generated:
                        try:
                            module = importlib.import_module(f".{module_name}", package="plots")
                            classes[module_name] = module
                        except (ImportError, AttributeError):
                            pass

        return classes
