import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy
from datacentertracesdatasets import loadtraces
import numpy as np
import pandas
from pandas import DataFrame
from tqdm import tqdm

from utils import get_ori_data_sample
from natsort import natsorted

from dtaidistance import dtw_ndim
from dtaidistance import dtw_visualisation
from dtaidistance import dtw



def create_figure(ori_column_values_array, generated_column_values, axis, name, path_to_save_metrics):
    plt.rcParams["figure.figsize"] = (18, 3)
    f, ax = plt.subplots(1)
    i = 1
    cycol = cycle('grcmk')

    for ori_column_values in ori_column_values_array:
        plt.plot(ori_column_values, c=next(cycol), label=f'Original_{i}', linewidth=1)
        i += 1

    plt.plot(generated_column_values, c="blue", label="Synthetic", linewidth=2)
    if axis is not None:
        plt.axis(axis)
    else:
        plt.xlim([0, len(ori_column_values_array[0])])

    plt.title(f'{name} original vs synthetic')
    plt.xlabel('time')
    plt.ylabel(name)
    ax.legend()
    plt.savefig(f'{path_to_save_metrics}{name}.pdf', format='pdf')
    plt.close('all')


def generate_figure_from_df(column_config_param, generated_data_sample_df, ori_data_sample, sample_filename, path):
    ori_data_sample_df = pandas.DataFrame(ori_data_sample, columns = [f'{column_name}_original' for column_name in generated_data_sample_df.columns])
    plt.rcParams["figure.figsize"] = (18, 3)
    plt.figure()
    ax=generated_data_sample_df.plot()
    ori_data_sample_df.plot(ax=ax, style='--', color='darkgrey')
    plt.xlim([0, generated_data_sample_df.shape[0]])
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('y_label')
    os.makedirs(f'{path}/all-columns/', exist_ok=True)
    plt.savefig(f'{path}/all-columns/{sample_filename}.pdf')
    plt.clf()
    plt.close('all')


def generate_dtw_figure(generated_data_sample_column, ori_data_sample_column, column_name, path_to_save_metrics, sample_filename):
    path = dtw.warping_path(ori_data_sample_column, generated_data_sample_column )
    dtw_visualisation.plot_warping(ori_data_sample_column, generated_data_sample_column, path, filename=f'{path_to_save_metrics}{column_name}-dtw_warping-{sample_filename}.pdf')
    pass

def create_usage_evolution(generated_data_sample, generated_data_sample_df, ori_data, ori_data_sample,
                           path_to_save_metrics, sample_filename,
                           dataset_info, generate_deltas):
    seq_len = len(ori_data_sample[:, 0])
    column_configs = dataset_info['column_config'].items()
    generate_figure_from_df(column_configs, generated_data_sample_df, ori_data_sample, sample_filename, path_to_save_metrics)
    for column_name, column_config in column_configs:
        index = column_config['column_index']
        path_to_save_metrics_column = path_to_save_metrics + '/' + column_name + '/'
        os.makedirs(path_to_save_metrics_column, exist_ok=True)
        generate_figures_by_column(index, column_name, generated_data_sample, ori_data, ori_data_sample,
                                   path_to_save_metrics_column, sample_filename, seq_len,
                                   dataset_info['timestamp_frequency_secs'], column_config, generate_deltas)
        generate_dtw_figure(generated_data_sample[:,index], ori_data_sample[:,index], column_name, path_to_save_metrics_column, sample_filename)


def generate_figures_by_column(column_number, column_name, generated_data_sample, ori_data, ori_data_sample,
                               path_to_save_metrics, sample_filename, seq_len, timestamp_frequency_secs, column_config, generate_deltas):
    if 'y_axis_min' in column_config and 'y_axis_max' in column_config:
        axis = [0, seq_len, column_config['y_axis_min'], column_config['y_axis_max']]
    else:
        axis = None

    create_figure(ori_column_values_array=[ori_data_sample[:, column_number]],
                  generated_column_values=generated_data_sample[:, column_number], axis=axis,
                  name=f'{column_name}_{sample_filename}', path_to_save_metrics=path_to_save_metrics)

    if generate_deltas == True:
        time_delta_minutes = [5, 10, 30, 60]
        for minutes in time_delta_minutes:
            generate_figures_grouped_by_minutes_various_ori_samples(minutes, column_number, column_name,
                                                                    generated_data_sample,
                                                                    ori_data, path_to_save_metrics, sample_filename,
                                                                    seq_len,
                                                                    timestamp_frequency_secs, 5)


def generate_figures_grouped_by_minutes_various_ori_samples(minutes, column_number, column_name, generated_data_sample,
                                                            ori_data, path_to_save_metrics, sample_filename, seq_len,
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
                  name=f'{column_name}_usage_delta_{round(minutes, 2)}min-{sample_filename}',
                  path_to_save_metrics=path_to_save_metrics)


def compute_grouped_delta_from_sample(column_number, minutes, data_sample, seq_len, timestamp_frequency_secs, ):
    sample_column = data_sample[:, column_number]
    sample_column_splitted = np.array_split(sample_column, seq_len // (minutes / (timestamp_frequency_secs / 60)))
    sample_column_mean = [np.mean(batch) for batch in sample_column_splitted]
    delta_sample_column = -np.diff(sample_column_mean)
    return delta_sample_column


def generate_inter_experiment_column_figure(df, filename_param, path, column_config_param=None):
    if ("y_axis_min" in column_config_param and "y_axis_max" in column_config_param):
        axis = [0, df.shape[0], column_config_param['y_axis_min'], column_config_param['y_axis_max']]
    else:
        axis = None
    colors = ['#ff0000', '#ff5700', '#ff8200', '#ffa500', '#ffc600', '#fff000', '#e0f500', '#bbf900', '#8efc00',
              '#48ff00', '#006313', '#008251', '#00a08e', '#00bdc9', '#00d8ff', '#3759ff', '#3340d0', '#2a29a4',
              '#1c147a', '#0b0053', '#37009b', '#52009b', '#67009a', '#79009a', '#890199', '#d200ff', '#d200c0',
              '#c2008a', '#a7005f', '#86003e']
    plt.figure()
    # df.plot(color=colors)
    df.plot(colormap=plt.get_cmap('RdYlGn'), figsize=(18, 3))
    if axis is not None:
        plt.axis(axis)
    else:
        plt.xlim([0, df.shape[0]])

    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('y_label')
    os.makedirs(f'{path}inter_experiment_figures/', exist_ok=True)
    plt.savefig(f'{path}inter_experiment_figures/{filename_param}.pdf')
    plt.clf()
    plt.close('all')


def generate_inter_experiment_figures(root_experiment_dir, experiments_dirs, args_params):
    dataset_info = loadtraces.get_dataset_info(trace_name=args_params.trace, trace_type=args_params.trace_type,
                                               stride_seconds=args_params.trace_timestep)
    data_frames = {}

    experiments_dirs = natsorted(experiments_dirs)

    for dir in experiments_dirs:
        data_frames[dir] = pandas.read_csv(f'{dir}/generated_data/sample_0.csv',
                                           names=list(dataset_info['column_config'].keys()))

    for column_name in tqdm(dataset_info['column_config'], desc='Generating inter-experiments figures'):
        plot_dataframe = DataFrame()
        for experiment_name, experiment_dataframe in data_frames.items():
            epoch_name = os.path.basename(os.path.normpath(experiment_name))
            plot_dataframe[epoch_name] = experiment_dataframe[column_name]
        generate_inter_experiment_column_figure(plot_dataframe, f'inter_experiment-{column_name}', root_experiment_dir,
                                                dataset_info['column_config'][column_name])
