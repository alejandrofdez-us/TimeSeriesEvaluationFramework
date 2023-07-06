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

        self.folder_path = os.path.dirname(os.path.abspath(__file__))
        self.figure_objects = self.__find_figures_in_directory(self.folder_path, figures_to_be_generated)

    @staticmethod
    def __find_figures_in_directory(folder_path, figures_to_be_generated):
        available_figures = PlotFactory.find_available_figures(folder_path)
        available_figures = {k: v for k, v in available_figures.items() if k in figures_to_be_generated}

        return available_figures

    @staticmethod
    def find_available_figures(folder_path):
        available_figures = find_available_classes(folder_path, Plot, "plots")
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
