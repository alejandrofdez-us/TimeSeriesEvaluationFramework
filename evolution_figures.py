import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas
from pandas import DataFrame

from utils import get_ori_data_sample, get_dataset_info


def create_figure(ori_column_values_array, generated_column_values, axis, name, path_to_save_metrics):
    plt.rcParams["figure.figsize"] = (18, 3)
    f, ax = plt.subplots(1)
    i = 1
    cycol = cycle('grcmk')

    for ori_column_values in ori_column_values_array:
        plt.plot(ori_column_values, c=next(cycol), label="Original_" + str(i), linewidth=1)
        i += 1

    plt.plot(generated_column_values, c="blue", label="Synthetic", linewidth=2)
    if axis is not None:
        plt.axis(axis)
    else:
        plt.xlim([0, len(ori_column_values_array[0])])

    plt.title(name + ' original vs sythetic')
    plt.xlabel('time')
    plt.ylabel(name)
    ax.legend()
    plt.savefig(path_to_save_metrics + name + '.pdf', format='pdf')
    plt.close()


def create_usage_evolution(generated_data_sample, ori_data, ori_data_sample, path_to_save_metrics, n_file,
                           dataset_info):
    seq_len = len(ori_data_sample[:, 0])
    column_configs = dataset_info['column_config'].items()
    for column_name, column_config in column_configs:
        index = column_config['column_index']
        path_to_save_metrics_column = path_to_save_metrics + '/' + column_name + '/'
        os.makedirs(path_to_save_metrics_column, exist_ok=True)
        generate_figures_by_column(index, column_name, generated_data_sample, ori_data, ori_data_sample,
                                   path_to_save_metrics_column, n_file, seq_len,
                                   dataset_info['timestamp_frequency_secs'], column_config)


def generate_figures_by_column(column_number, column_name, generated_data_sample, ori_data, ori_data_sample,
                               path_to_save_metrics, n_file, seq_len, timestamp_frequency_secs, column_config):
    path_to_save_metrics_for_file_number = path_to_save_metrics + str(n_file) + '-'
    if ("y_axis_min" in column_config and "y_axis_max" in column_config):
        axis = [0, seq_len, column_config['y_axis_min'], column_config['y_axis_max']]
    else:
        axis = None

    create_figure(ori_column_values_array=[ori_data_sample[:, column_number]],
                  generated_column_values=generated_data_sample[:, column_number], axis=axis,
                  name=column_name + '_usage', path_to_save_metrics=path_to_save_metrics_for_file_number)

    generate_figures_grouped_by_minutes_various_ori_samples(5, column_number, column_name, generated_data_sample,
                                                            ori_data, path_to_save_metrics_for_file_number, seq_len,
                                                            timestamp_frequency_secs, 5)
    generate_figures_grouped_by_minutes_various_ori_samples(10, column_number, column_name, generated_data_sample,
                                                            ori_data, path_to_save_metrics_for_file_number, seq_len,
                                                            timestamp_frequency_secs, 5)
    generate_figures_grouped_by_minutes_various_ori_samples(30, column_number, column_name, generated_data_sample,
                                                            ori_data, path_to_save_metrics_for_file_number, seq_len,
                                                            timestamp_frequency_secs, 5)
    generate_figures_grouped_by_minutes_various_ori_samples(60, column_number, column_name, generated_data_sample,
                                                            ori_data, path_to_save_metrics_for_file_number, seq_len,
                                                            timestamp_frequency_secs, 5)


def generate_figures_grouped_by_minutes_various_ori_samples(minutes, column_number, column_name, generated_data_sample,
                                                            ori_data, path_to_save_metrics, seq_len,
                                                            timestamp_frequency_secs, n_ori_samples=1):
    delta_ori_column_array = [
        compute_grouped_delta_from_sample(column_number, minutes, get_ori_data_sample(seq_len, ori_data), seq_len,
                                          timestamp_frequency_secs) for i in
        range(n_ori_samples)]

    delta_gen_column = compute_grouped_delta_from_sample(column_number, minutes, generated_data_sample, seq_len,
                                                         timestamp_frequency_secs)

    max_y_value = max(np.amax(delta_ori_column_array), np.amax(delta_gen_column))
    min_y_value = min(np.amin(delta_ori_column_array), np.amin(delta_gen_column))
    create_figure(ori_column_values_array=delta_ori_column_array, generated_column_values=delta_gen_column,
                  axis=[0, seq_len // (minutes / (timestamp_frequency_secs / 60)), min_y_value, max_y_value],
                  name=column_name + '_usage_delta_' + str(round(minutes, 2)) + 'min',
                  path_to_save_metrics=path_to_save_metrics)


def compute_grouped_delta_from_sample(column_number, minutes, data_sample, seq_len, timestamp_frequency_secs, ):
    sample_column = data_sample[:, column_number]
    sample_column_splitted = np.array_split(sample_column, seq_len // (minutes / (timestamp_frequency_secs / 60)))
    sample_column_mean = [np.mean(batch) for batch in sample_column_splitted]
    delta_sample_column = -np.diff(sample_column_mean)
    return delta_sample_column


def generate_inter_experiment_column_figure(df, filename_param, path, column_config_param=None):
    plt.rcParams["figure.figsize"] = (16, 9)
    colors = ['#ff0000', '#ff5700', '#ff8200', '#ffa500', '#ffc600', '#fff000', '#e0f500', '#bbf900', '#8efc00',
              '#48ff00', '#006313', '#008251', '#00a08e', '#00bdc9', '#00d8ff', '#3759ff', '#3340d0', '#2a29a4',
              '#1c147a', '#0b0053', '#37009b', '#52009b', '#67009a', '#79009a', '#890199', '#d200ff', '#d200c0',
              '#c2008a', '#a7005f', '#86003e']
    plt.figure()
    df.plot(color=colors)
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('y_label')
    os.makedirs(f'{path}inter_experiment_figures/', exist_ok=True)
    plt.savefig(f'{path}inter_experiment_figures/{filename_param}.pdf')
    print(f'Saved figure at: {path}/figures/{filename_param}.pdf')
    plt.clf()
    plt.close()


def generate_inter_experiment_figures(root_experiment_dir, experiments_dirs, trace_name):
    print('generate_inter_experiment_figures')
    dataset_info = get_dataset_info(trace_name)
    data_frames = {}

    if 'epoch' in experiments_dirs[0]:
        experiments_dirs = sorted(experiments_dirs, key=lambda fileName: int(fileName.split('.')[0].split('_')[1]))

    for dir in experiments_dirs:
        data_frames[dir] = pandas.read_csv(f'{dir}/generated_data/sample_0.csv',
                                           names=list(dataset_info['column_config'].keys()))

    for column_name in dataset_info['column_config']:
        plot_dataframe = DataFrame()
        for experiment_name, experiment_dataframe in data_frames.items():
            epoch_name = os.path.basename(os.path.normpath(experiment_name))
            plot_dataframe[epoch_name] = experiment_dataframe[column_name]
        generate_inter_experiment_column_figure(plot_dataframe, f'inter_experiment-{column_name}', root_experiment_dir)
