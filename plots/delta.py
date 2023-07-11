import random
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from plots.plot import Plot


class Delta(Plot):
    def __init__(self):
        super().__init__()
        self.seq_len = None
        self.ts_freq_secs = None
        self.n_ts1_samples_to_plot = None

    def initialize(self, core, filename):
        super().initialize(core, filename)
        self.seq_len = core.ts2_dict[filename].shape[0]
        self.ts_freq_secs = core.core_config.plot_config.timestamp_frequency_seconds
        self.n_ts1_samples_to_plot = 5
        self.time_magnitude, self.time_magnitude_name = self.compute_time_magnitude()

    def compute_time_magnitude(self):
        if self.ts_freq_secs < 60:
            time_magnitude = 1
            time_magnitude_name = 'seconds'
        elif self.ts_freq_secs < 60 * 60:
            time_magnitude = 60
            time_magnitude_name = 'minutes'
        elif self.ts_freq_secs < 60 * 60 * 24:
            time_magnitude = 60 * 60
            time_magnitude_name = 'hours'
        else:
            time_magnitude = 60 * 60 * 24
            time_magnitude_name = 'days'
        return time_magnitude, time_magnitude_name

    def generate_figures(self, core, filename):
        super().generate_figures(core, filename)

        time_intervals = [(self.ts_freq_secs / self.time_magnitude) * value for value in [2, 5, 10]]
        plot_array = []
        for column_index, column_name in enumerate(self.header_names):
            for time_interval in time_intervals:
                plot_array.append(
                    self.__generate_delta_grouped_by_interval(time_interval, column_index, column_name))
        return plot_array

    def __generate_delta_grouped_by_interval(self, time_interval, column_index, column_name):
        delta_ts1_column_array = [
            self.__compute_grouped_delta_from_sample(self.__get_random_ts1_sample(), column_index, time_interval) for _
            in
            range(self.n_ts1_samples_to_plot)]

        delta_ts2_column = self.__compute_grouped_delta_from_sample(self.ts2, column_index, time_interval)
        max_y_value = max(np.amax(delta_ts1_column_array), np.amax(delta_ts2_column))
        min_y_value = min(np.amin(delta_ts1_column_array), np.amin(delta_ts2_column))
        return self.__create_figure(ts1_column_values_array=delta_ts1_column_array,
                                    ts2_column_values=delta_ts2_column, column_name=column_name,
                                    axis=[0, len(delta_ts2_column) - 1, min_y_value, max_y_value],
                                    time_interval=time_interval)

    def __get_random_ts1_sample(self):
        return self.ts1_windows[np.random.randint(0, len(self.ts1_windows))]

    def __compute_grouped_delta_from_sample(self, data_sample, column_number, time_interval):
        sample_column = data_sample[:, column_number]
        seq_len = data_sample.shape[0]
        sample_column_split = np.array_split(sample_column,
                                             seq_len // (time_interval / (self.ts_freq_secs / self.time_magnitude)))
        sample_column_mean = [np.mean(batch) for batch in sample_column_split]
        delta_sample_column = -np.diff(sample_column_mean)
        return delta_sample_column

    def __create_figure(self, ts1_column_values_array, ts2_column_values, column_name, axis, time_interval):
        plt.rcParams["figure.figsize"] = (18, 3)
        fig, ax = plt.subplots(1)
        i = 1
        cycle_colors = cycle('grcmk')
        for ts1_column_values in ts1_column_values_array:
            plt.plot(ts1_column_values, c=next(cycle_colors), label=f"TS_1_sample_{i}", linewidth=1)
            i += 1
        plt.plot(ts2_column_values, c="blue", label="TS_2", linewidth=3)
        plt.axis(axis)
        plt.title(f'{column_name}_TS_1_vs_TS_2_(grouped_by_{int(time_interval)}_{self.time_magnitude_name})')
        plt.xlabel('time')
        plt.ylabel(column_name)
        ax.legend()
        plt.close(fig)
        plot_tuple = (fig, ax)
        return plot_tuple
