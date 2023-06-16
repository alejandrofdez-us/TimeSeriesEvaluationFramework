import numpy as np
import pandas as pd
import csv
import re
import os
import random

from dtaidistance import dtw_ndim

def csv_has_header(filename, ts_delimiter, has_header):
    if has_header:
        header = np.genfromtxt(filename, delimiter=ts_delimiter, names=has_header, max_rows=1, dtype=str).dtype.names

        if header_has_numeric(header):
            raise ValueError("Header must not contain numeric values.")

    else:
        header = np.loadtxt(filename, delimiter=ts_delimiter, max_rows=1)
        header = ["column-"+str(i) for i in range(len(header))]

    return header

def header_has_numeric(header):
    pattern = r'^[-+]?\d*\.?\d+$'
    for column in header:
        if re.match(pattern, column):
            return True
    return False

def detect_line_delimiter(filename):
    with open(filename, "r", newline="") as file:
        ts_delimiter = csv.Sniffer().sniff(file.readline()).delimiter

    return ts_delimiter

def load_ts_from_csv(filename, has_header=None):
    ts_delimiter = detect_line_delimiter(filename)

    header = csv_has_header(filename, ts_delimiter, has_header)
    skiprows = 1 if has_header else 0

    return np.loadtxt(filename, delimiter=ts_delimiter, skiprows=skiprows), header, ts_delimiter

def update_figures_arguments(time_series_1, time_series_2, header, figures_to_be_generated, ts_freq_secs):
    args = {"ts1" : time_series_1, "ts2" : time_series_2, "header" : header}

    if "tsne" in figures_to_be_generated or "pca" in figures_to_be_generated:
        args.update(tsne_pca_preprocess(time_series_1, time_series_2))

    if "deltas" in figures_to_be_generated:
        args.update(deltas_preprocess(time_series_1, ts_freq_secs))
    
    if "evolution" in figures_to_be_generated:
        args.update(evolution_preprocess(time_series_1, time_series_2, header))

    return args

def deltas_preprocess(ts1, ts_freq_secs):

    args = {"seq_len" : len(ts1[:, 0]), "ts_freq_secs" : ts_freq_secs, "n_ori_samples" : 1}

    return args

def evolution_preprocess(ts1, ts2, header):

    generated_data_sample_df = pd.DataFrame(ts2, columns=header)
    args = {"seq_len" : len(ts1[:, 0]), "ori_data_sample" : ts1, "generated_data_sample" : ts2,
                "generated_data_sample_df" : generated_data_sample_df}

    return args

def tsne_pca_preprocess (ts1, ts2):
    generated_data = []
    n_samples = 0
    seq_len = 0

    n_samples = 1
    seq_len = len(ts2)
    generated_data.append(ts2)

    ori_data_for_visualization = seq_cut_and_mix(ts1, seq_len)

    args = {"ori_data" : ori_data_for_visualization, "gen_data" : generated_data, "n_samples" : n_samples}
    plot_args = plot_preprocess(args)
    
    return plot_args
        
def seq_cut_and_mix (ori_data, seq_len):
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len + 1):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data

def plot_preprocess (args):
    # Analysis sample size (for faster computation)
    anal_sample_no = min([args["n_samples"], len(args["ori_data"])])
    idx = np.random.permutation(args["n_samples"])[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(args["ori_data"])
    generated_data = np.asarray(args["gen_data"])

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    _, seq_len, _ = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for _ in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]
    
    args.pop("ori_data")
    args.pop("gen_data")

    args.update({"anal_sample_no" : anal_sample_no, "prep_data": prep_data, "prep_data_hat" : prep_data_hat, "colors" : colors})

    return args

def get_ori_data_sample(seq_len, ori_data):
    if len(ori_data) > seq_len:
        ori_data_sample_start = random.randrange(0, len(ori_data) - seq_len)
    else:  # seq_len is the full ori_data_length
        ori_data_sample_start = 0

    ori_data_sample_end = ori_data_sample_start + seq_len
    ori_data_sample = ori_data[ori_data_sample_start:ori_data_sample_end]
    return ori_data_sample

def get_most_similar_ori_data_sample(ori_data_windows_numpy, generated_data_sample):
    minimum_dtw = float('inf')
    most_similar_sample = []
    for ori_data_sample in ori_data_windows_numpy:
        # TODO: Preguntamos por el parametro que define la metrica a usar para calcular la ventana mas parecida: mmd, dtw por defecto, ks, cc...
        # TODO: Definimos un argumento opcional para la metrica a usar
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
        ori_data_windows_numpy = np.array(
            [ori_data_df[start_index:start_index + seq_len] for start_index in start_sequence_range])
    return ori_data_windows_numpy

def normalize_start_time_to_zero(sample):
    timestamp_column = sample[:, 0]
    min_timestamp = np.min(timestamp_column)
    normalized_timestamp_column = timestamp_column - min_timestamp
    sample[:, 0] = normalized_timestamp_column
    return sample

def extract_experiment_parameters(saved_experiment_parameters):
    saved_experiment_parameters_dict = dict(
        item.split("=") for item in re.split(', (?![^\[]*\])',
                                             saved_experiment_parameters.replace('Namespace(', '').replace(
                                                 'Parameters(', '').replace(')', '').replace('\n', '')))
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
    for metric_name, avg_result in avg_results.items():
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
                print(
                    f'Previous csv result could not be retrieved from {dir_name}/time-series-framework-metrics.csv. '
                    f'Details: {e}')
