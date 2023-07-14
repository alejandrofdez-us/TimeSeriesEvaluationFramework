from dtaidistance import dtw_visualisation
from dtaidistance import dtw
from matplotlib import pyplot as plt
from plots.plot import Plot


class Dtw(Plot):
    def compute(self, similarity_ts, filename):
        super().compute(similarity_ts, filename)
        figures = []
        for column_name, index in zip(self.header_names, range(len(self.header_names))):
            figures.append(
                self.__generate_plot(
                    self.ts1[:, index], self.ts2[:, index], column_name
                )
            )
        return figures

    def __generate_plot(self, ts1_column, ts2_column, column_name):
        super()._init_plot()
        path = dtw.warping_path(ts1_column, ts2_column)
        fig, axes = dtw_visualisation.plot_warping(ts1_column, ts2_column, path)
        plt.xlim(left=0, right=len(ts1_column) - 1)
        axes[0].set_title(f'DTW_{column_name}')
        axes[0].legend(['TS_1'], loc='center right')
        axes[1].legend(['TS_2'], loc='center right')
        plt.close('all')
        return fig, axes
