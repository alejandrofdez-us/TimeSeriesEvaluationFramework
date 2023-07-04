from dtaidistance import dtw_visualisation
from dtaidistance import dtw
from matplotlib import pyplot as plt

from plots.plot import Plot

class Dtw(Plot):
    def generate_figures(self, args):
        plot_array = self.__generate_dtw_figures(args)
        return plot_array

    def __generate_dtw_figures(self, args):
        figures = []
        for column, index in zip(args["header"], range(len(args["header"]))):
            figures.append(
                self.__generate_dtw_figure(
                    args["ts1"][:, index], args["ts2"][:, index], column
                )
            )
        return figures
    
    def __generate_dtw_figure(self, time_series_1_column, time_series_2_column, column):
        path = dtw.warping_path(time_series_1_column, time_series_2_column)
        figure = dtw_visualisation.plot_warping(
            time_series_1_column, time_series_2_column, path
        )

        figure[0].axes[0].set_title(f"DTW_{column}")
        plt.close()

        return figure
