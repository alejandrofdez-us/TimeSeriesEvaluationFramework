from itertools import cycle
import matplotlib.pyplot as plt
from helper import get_ori_data_sample
import numpy as np

from plots.plot import Plot

class Deltas(Plot):
    def generate_figures(self, args):
        plot_array = self.__generate_deltas_figures(args)
        return plot_array

    def __generate_deltas_figures(self, args):

        plot_array = []
        for index, column in enumerate(args["header"]):
            time_delta_minutes = [2, 5, 10]
            time_delta_minutes = [(args["ts_freq_secs"]/60) * value for value in time_delta_minutes]
            
            for minutes in time_delta_minutes:
                plot_array.append(self.__generate_figures_grouped_by_minutes_various_ori_samples(minutes, index, column, args["ts1"], args["ts2"], 
                                                                        args["seq_len"], args["ts_freq_secs"], args["n_ori_samples"]))
        
        return plot_array

    def __generate_figures_grouped_by_minutes_various_ori_samples(self, minutes, column_number, column_name, ori_data, generated_data_sample, 
                                                                seq_len, ts_freq_secs, n_ori_samples):
        delta_ori_column_array = [
            self.__compute_grouped_delta_from_sample(column_number, minutes, get_ori_data_sample(seq_len, ori_data), seq_len,
                                            ts_freq_secs) for _ in range(n_ori_samples)]

        delta_gen_column = self.__compute_grouped_delta_from_sample(column_number, minutes, generated_data_sample, seq_len,
                                                            ts_freq_secs)

        max_y_value = max(np.amax(delta_ori_column_array), np.amax(delta_gen_column))
        min_y_value = min(np.amin(delta_ori_column_array), np.amin(delta_gen_column))
        return self.__create_figure(ori_column_values_array=delta_ori_column_array, generated_column_values=delta_gen_column, column_name=column_name,
                    axis=[0, seq_len // (minutes / (ts_freq_secs / 60)), min_y_value, max_y_value], minutes=minutes)
        
    def __compute_grouped_delta_from_sample(self, column_number, minutes, data_sample, seq_len, ts_freq_secs):
        sample_column = data_sample[:, column_number]
        sample_column_splitted = np.array_split(sample_column, seq_len // (minutes / (ts_freq_secs / 60)))
        sample_column_mean = [np.mean(batch) for batch in sample_column_splitted]
        delta_sample_column = -np.diff(sample_column_mean)
        return delta_sample_column

    def __create_figure(self, ori_column_values_array, generated_column_values, column_name, axis, minutes):
        plt.rcParams["figure.figsize"] = (18, 3)
        fig, ax = plt.subplots(1)
        i = 1
        cycol = cycle('grcmk')

        for ori_column_values in ori_column_values_array:
            plt.plot(ori_column_values, c=next(cycol), label=f'TS_1', linewidth=1)
            i += 1

        plt.plot(generated_column_values, c="blue", label="TS_2", linewidth=2)
        if axis is not None:
            plt.axis(axis)
        else:
            plt.xlim([0, len(ori_column_values_array[0])])

        plt.title(f'{column_name}_TS_1_vs_TS_2_(grouped_by_{int(minutes)}_minutes)')
        plt.xlabel('time')
        plt.ylabel(column_name)
        ax.legend()

        plot_tuple = (fig, ax)

        return plot_tuple