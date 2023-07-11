from itertools import cycle
import matplotlib.pyplot as plt
import pandas
from plots.plot import Plot


class TwoDimensions(Plot):
    def __init__(self):
        super().__init__()
        self.seq_len = None
        self.ts2_df = None

    def initialize(self, core, filename):
        super().initialize(core, filename)
        self.ts2_df = pandas.DataFrame(core.ts2_dict[filename], columns=core.header_names)
        self.seq_len = core.ts2_dict[filename].shape[0]

    def get_name(self):
        return "2d"

    def generate_figures(self, core, filename):
        super().generate_figures(core, filename)
        plot_array = []
        plot_array.append(self.__generate_figure_from_df())
        for index, column in enumerate(self.header_names):
            plot_array.append(self.__generate_figures_by_column(index, column))
        return plot_array

    def __generate_figure_from_df(self):
        ts_sample_df = pandas.DataFrame(self.ts1, columns=[f'{column_name}_TS_1' for column_name in
                                                           self.ts2_df.columns])
        fig, ax = plt.subplots(1)
        ts_sample_df.plot(ax=ax, color='darkgrey')
        new_column_names = {col: col + "_TS_2" for col in self.ts2_df.columns}
        self.ts2_df = self.ts2_df.rename(columns=new_column_names)
        self.ts2_df.plot(ax=ax, style='--')
        plt.xlim([0, self.ts2_df.shape[0]])
        plt.title("all_columns")
        plt.legend(loc='best')
        plt.xlabel('time')
        plt.ylabel('y_label')
        fig.clf()
        plt.close()
        return fig, ax

    def __generate_figures_by_column(self, column_number, column_name):
        return self.__create_figure(ts1_column_values_array=[self.ts1[:, column_number]],
                                    ts2_column_values=self.ts2[:, column_number], column_name=column_name)

    def __create_figure(self, ts1_column_values_array, ts2_column_values, column_name, axis=None):
        fig, ax = plt.subplots(1)
        i = 1
        cycol = cycle('grcmk')

        for ts1_column_values in ts1_column_values_array:
            plt.plot(ts1_column_values, c=next(cycol), label="TS_1", linewidth=1)
            i += 1

        plt.plot(ts2_column_values, c="blue", label="TS_2", linewidth=2)
        if axis is not None:
            plt.axis(axis)
        else:
            plt.xlim([0, len(ts1_column_values_array[0])])

        plt.title(f'{column_name}_TS_1_vs_TS_2')
        plt.xlabel('time')
        plt.ylabel(column_name)
        ax.legend()
        fig.clf()
        plt.close()
        return fig, ax
