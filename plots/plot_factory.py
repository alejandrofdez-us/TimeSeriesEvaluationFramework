import os
from plots.plot import Plot
from dynamic_import_helper import find_available_classes


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class PlotFactory(metaclass=Singleton):
    def __init__(self, figure_names_to_be_generated):
        self.plots_to_be_generated = self.__get_plots_to_be_generated(figure_names_to_be_generated)
        self.figures_requires_all_samples = self.__get_figures_that_requires_all_samples(figure_names_to_be_generated)

    @staticmethod
    def __get_plots_to_be_generated(figure_names_to_be_generated):
        available_plots = PlotFactory.find_available_figures()
        plots_to_be_generated = {plot_name: plot for plot_name, plot in available_plots.items() if
                                 plot_name in figure_names_to_be_generated}
        return plots_to_be_generated
    
    @staticmethod
    def find_available_figures():
        return find_available_classes(os.path.dirname(os.path.abspath(__file__)), Plot, 'plots')

    @staticmethod
    def get_instance(figure_names_to_be_generated=None):
        if not hasattr(PlotFactory, "_instance"):
            if figure_names_to_be_generated is None:
                figure_names_to_be_generated = ['2d', 'deltas', 'dtw', 'pca', 'tsne']
            PlotFactory._instance = PlotFactory(figure_names_to_be_generated)
        return PlotFactory._instance

    def __get_figures_that_requires_all_samples(self, figures_to_be_generated):
        return [figure_name for figure_name in figures_to_be_generated if
                self.__figure_requires_all_samples(figure_name)]

    def __figure_requires_all_samples(self, figure_name):
        return self.plots_to_be_generated[figure_name].requires_all_samples()
