import os
import random
import re

from dtaidistance import dtw_ndim
import numpy as np
from tqdm import tqdm


def get_ori_data_sample(seq_len, ori_data):
    if len(ori_data) > seq_len:
        ori_data_sample_start = random.randrange(0, len(ori_data) - seq_len)
    else: # seq_len is the full ori_data_length
        ori_data_sample_start = 0

    ori_data_sample_end = ori_data_sample_start + seq_len
    ori_data_sample = ori_data[ori_data_sample_start:ori_data_sample_end]
    return ori_data_sample

def get_most_similar_ori_data_sample(ori_data_windows_numpy, generated_data_sample):
    minimum_dtw = float('inf')
    most_similar_sample = []
    for ori_data_sample in ori_data_windows_numpy:
        current_distance = dtw_ndim.distance_fast(ori_data_sample, generated_data_sample)
        if current_distance < minimum_dtw:
            minimum_dtw = current_distance
            most_similar_sample = ori_data_sample

    return most_similar_sample, minimum_dtw

def split_ori_data_strided(ori_data_df, seq_len, stride):
    assert seq_len <= ori_data_df.shape[0], 'seq_len cannot be greater than the original dataset length'
    if seq_len == ori_data_df.shape[0]:
        ori_data_windows_numpy = np.array([ori_data_df])
    else:
        start_sequence_range = list(range(0, ori_data_df.shape[0] - seq_len, stride))
        ori_data_windows_numpy = np.array([ori_data_df[start_index:start_index+seq_len] for start_index in start_sequence_range])
    return ori_data_windows_numpy

def normalize_start_time_to_zero(sample):
    timestamp_column = sample[:, 0]
    min_timestamp = np.min(timestamp_column)
    normalized_timestamp_column = timestamp_column - min_timestamp
    sample[:, 0] = normalized_timestamp_column
    return sample


def extract_experiment_parameters(saved_experiment_parameters):
    saved_experiment_parameters_dict = dict(
        item.split("=") for item in re.split(', (?![^\[]*\])', saved_experiment_parameters.replace('Namespace(', '').replace('Parameters(', '').replace(')', '').replace('\n', '')))
    parameters_values = ''
    parameters_keys = ''
    for parameter_value in saved_experiment_parameters_dict.values():
        parameters_values += parameter_value + ';'
    for parameter_key in saved_experiment_parameters_dict.keys():
        parameters_keys += parameter_key + ';'
    return parameters_keys, parameters_values, saved_experiment_parameters_dict


def save_metrics(avg_results, metrics_results, path_to_save_metrics, saved_experiment_parameters, saved_metrics):
    path_to_save_metrics = os.path.dirname(os.path.dirname(path_to_save_metrics))

    with open(f'{path_to_save_metrics}/time-series-framework-metrics.txt', 'w') as f:
        f.write(saved_experiment_parameters + '\n\n')
        f.write(saved_metrics + '\n\n')
        f.write(repr(avg_results) + '\n')
        computed_metrics, metrics_values = results_for_excel(avg_results)
        f.write(
            'Results of the following metrics: ' + computed_metrics + ' in spanish locale Excel format:' + '\n' + metrics_values + '\n')
        f.write(repr(metrics_results))

    experiment_results_csv_filename = f'{path_to_save_metrics}/time-series-framework-metrics.csv'
    parameters_keys, parameters_values, _ = extract_experiment_parameters(saved_experiment_parameters)
    print_csv_header(experiment_results_csv_filename, parameters_keys, computed_metrics)
    print_csv_result_row(path_to_save_metrics, experiment_results_csv_filename, metrics_values,
                         parameters_values)

    return computed_metrics, metrics_values


def results_for_excel(avg_results):
    metrics_values = ''
    computed_metrics = ''
    for metric_name,avg_result in avg_results.items():
        computed_metrics += metric_name + ';'
        metrics_values += str(avg_result).replace('.', ',') + ';'

    return computed_metrics, metrics_values


def print_csv_result_row(experiment_dir_name, experiment_results_file_name, metrics_values, parameters_values):
    with open(experiment_results_file_name, 'a') as f:
        f.write(experiment_dir_name + ';' + parameters_values + metrics_values + '\n')


def print_csv_header(experiment_results_file_name, parameters_keys, saved_metrics):
    with open(experiment_results_file_name, 'w') as f:
        f.write('experiment_dir_name;' + parameters_keys + saved_metrics + '\n')


def print_previously_computed_experiments_metrics(experiment_directories_previously_computed,
                                                  experiment_results_file_name):
    with open(experiment_results_file_name, 'a') as composed_results_file:
        for dir_name in experiment_directories_previously_computed:
            try:
                with open(f'{dir_name}/time-series-framework-metrics.csv', 'r') as previously_computed_metrics:
                    results_row = previously_computed_metrics.readlines()[1]
                    composed_results_file.write(results_row)
            except Exception as e:
                print(f'Previous csv result could not be retrieved from {dir_name}/time-series-framework-metrics.csv. Details: {e}')