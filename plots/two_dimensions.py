from itertools import cycle
import matplotlib.pyplot as plt
import pandas
from plots.plot import Plot


class TwoDimensions(Plot):
    def __init__(self):
        super().__init__()
        self.seq_len = None
        self.ts2_df = None
        self.ts1_df = None

    def initialize(self, core, filename):
        super().initialize(core, filename)
        self.ts1_df = pandas.DataFrame(self.ts1, columns=[f'{column_name}_TS_1' for column_name in
                                                          core.header_names])
        self.ts2_df = pandas.DataFrame(core.ts2_dict[filename], columns=[f'{column_name}_TS_2' for column_name in
                                                                         core.header_names])
        self.seq_len = core.ts2_dict[filename].shape[0]

    def get_name(self):
        return "2d"

    def compute(self, core, filename):
        super().compute(core, filename)
        plot_array = [self.__generate_figure_from_df()]
        for index, column in enumerate(self.header_names):
            plot_array.append(self.__generate_figures_by_column(index, column))
        return plot_array

    def __generate_figure_from_df(self):
        plt.rcParams["figure.figsize"] = self.fig_size
        fig, ax = plt.subplots(1)
        self.ts1_df.plot(ax=ax, style='--', colormap='Qualitative')
        plt.gca().set_prop_cycle(None)
        self.ts2_df.plot(ax=ax, colormap='Qualitative')
        plt.xlim([0, self.ts2_df.shape[0]])
        plt.title("all_columns")
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('y_label')
        plt.close("all")
        return fig, ax

    def __generate_figures_by_column(self, column_number, column_name):
        return self.__create_figure(ts1_column_values_array=[self.ts1[:, column_number]],
                                    ts2_column_values=self.ts2[:, column_number], column_name=column_name)

    def __create_figure(self, ts1_column_values_array, ts2_column_values, column_name, axis=None):
        plt.rcParams["figure.figsize"] = self.fig_size
        fig, ax = plt.subplots(1)
        cycol = cycle('grcmk')
        for ts1_column_values in ts1_column_values_array:
            plt.plot(ts1_column_values, c=next(cycol), label="TS_1", linewidth=1)
        plt.plot(ts2_column_values, c="blue", label="TS_2", linewidth=2)
        if axis is not None:  # FIXME: Ejes necesarios?
            plt.axis(axis)
        else:
            plt.xlim([0, len(ts1_column_values_array[0])])
        plt.title(f'{column_name}_TS_1_vs_TS_2')
        plt.xlabel('time')
        plt.ylabel(column_name)
        ax.legend()
        plt.close("all")
        return fig, ax
