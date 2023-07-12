from dtaidistance import dtw_visualisation
from dtaidistance import dtw
from matplotlib import pyplot as plt
from plots.plot import Plot


class Dtw(Plot):
    def compute(self, core, filename):
        super().compute(core, filename)
        figures = []
        for column_name, index in zip(self.header_names, range(len(self.header_names))):
            figures.append(
                self.__generate_dtw_figure(
                    self.ts1[:, index], self.ts2[:, index], column_name
                )
            )
        return figures

    def __generate_dtw_figure(self, time_series_1_column, time_series_2_column, column_name):
        plt.rcParams["figure.figsize"] = self.fig_size
        path = dtw.warping_path(time_series_1_column, time_series_2_column)
        fig, axes = dtw_visualisation.plot_warping(
            time_series_1_column, time_series_2_column, path
        )
        fig.axes[0].set_title(f"DTW_{column_name}")
        plt.close("all")
        return fig, axes
