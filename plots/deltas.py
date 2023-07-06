from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

from plot_helper import get_random_time_series_sample
from plots.plot import Plot

class Deltas(Plot):
    def generate_figures(self, args):

        plot_array = []
        for index, column in enumerate(args["header"]):
            time_delta_minutes = [2, 5, 10]
            time_delta_minutes = [(args["ts_freq_secs"]/60) * value for value in time_delta_minutes]

            for minutes in time_delta_minutes:
                plot_array.append(self.__generate_figures_grouped_by_minutes_various_ts_samples(minutes, index, column, args["ts1"], args["ts2"], 
                                                                        args["seq_len"], args["ts_freq_secs"], args["n_ts1_samples"]))

        return plot_array

    def __generate_figures_grouped_by_minutes_various_ts_samples(self, minutes, column_number, column_name, ts1, generated_data_sample, 
                                                                seq_len, ts_freq_secs, n_ts1_samples):
        delta_ts1_column_array = [
            self.__compute_grouped_delta_from_sample(column_number, minutes, get_random_time_series_sample(seq_len, ts1), seq_len,
                                            ts_freq_secs) for _ in range(n_ts1_samples)]

        delta_gen_column = self.__compute_grouped_delta_from_sample(column_number, minutes, generated_data_sample, seq_len,
                                                            ts_freq_secs)

        max_y_value = max(np.amax(delta_ts1_column_array), np.amax(delta_gen_column))
        min_y_value = min(np.amin(delta_ts1_column_array), np.amin(delta_gen_column))
        return self.__create_figure(ts1_column_values_array=delta_ts1_column_array, generated_column_values=delta_gen_column, column_name=column_name,
                    axis=[0, len(delta_ts1_column_array[0])-1, min_y_value, max_y_value], minutes=minutes)

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
            plt.plot(ts1_column_values, c=next(cycol), label=f'TS_1', linewidth=1)
            i += 1

        plt.plot(generated_column_values, c="blue", label="TS_2", linewidth=2)

        plt.axis(axis)

        plt.title(f'{column_name}_TS_1_vs_TS_2_(grouped_by_{int(minutes)}_minutes)')
        plt.xlabel('time')
        plt.ylabel(column_name)
        ax.legend()

        plot_tuple = (fig, ax)

        plt.close(fig)

        return plot_tuple
