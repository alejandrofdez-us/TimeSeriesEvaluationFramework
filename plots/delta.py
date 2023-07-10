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
    
    def generate_figures(self, core, filename):
        self.initialize(core, filename)

        plot_array = []
        for index, column in enumerate(self.header_names):
            time_delta_minutes = [2, 5, 10]
            time_delta_minutes = [(self.ts_freq_secs / 60) * value for value in time_delta_minutes]

            for minutes in time_delta_minutes:
                plot_array.append(
                    self.__generate_figures_grouped_by_minutes_various_ts_samples(minutes, index, column, self.ts1_windows,
                                                                                  self.ts2,
                                                                                  core.ts2_dict[filename].shape[0], self.ts_freq_secs,
                                                                                  self.n_ts1_samples_to_plot))
        return plot_array

    def __generate_figures_grouped_by_minutes_various_ts_samples(self, minutes, column_number, column_name, ts1_windows,
                                                                 generated_data_sample,
                                                                 seq_len, ts_freq_secs, n_ts1_samples):
        delta_ts1_column_array = [
            self.__compute_grouped_delta_from_sample(column_number, minutes,
                                                     self.get_random_time_series_sample(), seq_len,
                                                     ts_freq_secs) for _ in range(n_ts1_samples)]

        delta_gen_column = self.__compute_grouped_delta_from_sample(column_number, minutes, generated_data_sample,
                                                                    seq_len,
                                                                    ts_freq_secs)

        max_y_value = max(np.amax(delta_ts1_column_array), np.amax(delta_gen_column))
        min_y_value = min(np.amin(delta_ts1_column_array), np.amin(delta_gen_column))
        return self.__create_figure(ts1_column_values_array=delta_ts1_column_array,
                                    generated_column_values=delta_gen_column, column_name=column_name,
                                    axis=[0, len(delta_ts1_column_array[0]) - 1, min_y_value, max_y_value],
                                    minutes=minutes)
    
    def get_random_time_series_sample(self):
        if len(self.ts1_windows) > self.seq_len:
            ts_sample_start = random.randrange(0, len(self.ts1_windows) - self.seq_len)
        else:
            ts_sample_start = 0
        ts_sample_end = ts_sample_start + self.seq_len
        ts_sample = self.ts1_windows[ts_sample_start:ts_sample_end]
        return ts_sample

    def __compute_grouped_delta_from_sample(self, column_number, minutes, data_sample, seq_len, ts_freq_secs):
        sample_column = data_sample[:, column_number]
        sample_column_splitted = np.array_split(sample_column, seq_len // (minutes / (ts_freq_secs / 60)))
        sample_column_mean = [np.mean(batch) for batch in sample_column_splitted]
        delta_sample_column = -np.diff(sample_column_mean)
        return delta_sample_column

    def __create_figure(self, ts1_column_values_array, generated_column_values, column_name, axis, minutes):
        plt.rcParams["figure.figsize"] = (18, 3)
        fig, ax = plt.subplots(1)
        i = 1
        cycol = cycle('grcmk')

        for ts1_column_values in ts1_column_values_array:
            plt.plot(ts1_column_values, c=next(cycol), label=f"TS_1_sample_{i}", linewidth=1)
            i += 1

        plt.plot(generated_column_values, c="blue", label="TS_2", linewidth=3)
        plt.axis(axis)
        plt.title(f'{column_name}_TS_1_vs_TS_2_(grouped_by_{int(minutes)}_minutes)')
        plt.xlabel('time')
        plt.ylabel(column_name)
        ax.legend()
        plt.close(fig)
        plot_tuple = (fig, ax)
        return plot_tuple
