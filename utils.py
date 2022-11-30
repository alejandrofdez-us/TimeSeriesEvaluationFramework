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
    progress_bar_search = tqdm(ori_data_windows_numpy, desc='Searching minimum dtw distance from original data', colour='yellow', position=2, leave=False)
    for ori_data_sample in progress_bar_search:
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


def save_metrics(avg_results, metrics_results, path_to_save_metrics, saved_experiments_parameters, saved_metrics):
    _, _, parameters_dict = extract_experiment_parameters(saved_experiments_parameters)
    if 'data_name' in parameters_dict:
        data_name = parameters_dict['data_name']
    elif 'trace' in parameters_dict:
        data_name = parameters_dict['trace']
    else:
        data_name = 'not_found'
    if 'iteration' in parameters_dict:
        epoch_value = parameters_dict['iteration']
        epoch_name = 'iterations'
    elif 'epochs' in parameters_dict:
        epoch_value = parameters_dict['epochs']
        epoch_name = 'epochs'
    elif 'gan_epochs' in parameters_dict:
        epoch_value = parameters_dict['gan_epochs']
        epoch_name = 'gan_epochs'
    else:
        epoch_value = '_'
        epoch_name = 'no_epoch_value_found'


    seq_len = parameters_dict['seq_len']
    with open(f'{path_to_save_metrics}/metrics-{data_name}-{epoch_name}-{epoch_value}-seq_len-{seq_len}.txt', 'w') as f:

        f.write(saved_experiments_parameters + '\n\n')
        f.write(saved_metrics + '\n\n')
        f.write(repr(avg_results) + '\n')
        computed_metrics, metrics_values = results_for_excel(avg_results)
        f.write(
            'Results of the following metrics: ' + computed_metrics + ' in spanish locale Excel format:' + '\n' + metrics_values + '\n')
        f.write(repr(metrics_results))
    #print("Metrics saved in file", f.name)
    return computed_metrics, metrics_values


def results_for_excel(avg_results):
    metrics_values = ''
    computed_metrics = ''
    for metric_name,avg_result in avg_results.items():
        computed_metrics += metric_name + ';'
        metrics_values += str(avg_result).replace('.', ',') + ';'

    return computed_metrics, metrics_values
