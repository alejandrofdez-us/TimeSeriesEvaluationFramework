import os

from plots.plot import Plot
from reader import find_available_classes

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
        self.figures_requires_all_samples = PlotFactory.get_figures_that_requires_all_samples(figures_to_be_generated)
        self.computed_figures_requires_all_samples= []

        self.folder_path = os.path.dirname(os.path.abspath(__file__))
        self.figure_classes = self.__find_figures_in_directory(self.folder_path, figures_to_be_generated)

    def create_figure(self, figure_to_be_computed):

        if figure_to_be_computed in self.figure_classes.keys():
            figure_object = self.figure_classes[figure_to_be_computed]
            return figure_object
        else:
            raise ValueError('Invalid metric name')

    @staticmethod    
    def __find_figures_in_directory(folder_path, figures_to_be_generated):
        available_figures = find_available_classes(folder_path, Plot, "plots")
        available_figures = {k: v for k, v in available_figures.items() if k in figures_to_be_generated}

        return available_figures

    @staticmethod
    def get_figures_that_requires_all_samples(figures_to_be_generated):
        folder_path = os.path.dirname(os.path.abspath(__file__))
        figure_classes = PlotFactory.__find_figures_in_directory(folder_path, figures_to_be_generated)
        figures_that_requires_all_samples = []
        for figure_name, figure_object in figure_classes.items():
            if figure_object.requires_all_samples():
                figures_that_requires_all_samples.append(figure_name)
        return figures_that_requires_all_samples
