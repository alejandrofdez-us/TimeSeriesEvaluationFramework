import argparse
import contextlib
import fnmatch
import io
import os
import re
import statistics
import sys
import traceback
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import scipy
from datacentertracesdatasets import loadtraces
from dtaidistance import dtw_ndim
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from tqdm import tqdm

from evolution_figures import create_usage_evolution
from metrics.kl import KLDivergenceUnivariate
from metrics.kl import KLdivergence, JSdistance, JSdistanceMultivariate
from metrics.mmd import mmd_rbf
from metrics.visualization_metrics import visualization
from utils import get_ori_data_sample


def main(args_params):
    if args_params.recursive == 'true':
        root_dir = args_params.experiment_dir
        experiment_results_file_name = root_dir + 'experiments_metrics-' + datetime.now().strftime(
            "%j-%H-%M-%S") + '.csv'
        experiment_directories = []
        for subdir, dirs, files in os.walk(root_dir):
            if 'generated_data' in dirs:
                experiment_directories.append(subdir)
        is_header_printed = False
        progress_bar = tqdm(experiment_directories, colour="green")
        for dir_name in progress_bar:
            args_params.experiment_dir = dir_name
            try:
                progress_bar.set_description("Computing metrics for directory" + dir_name)
                #print("Computing metrics for directory ", dir_name)
                saved_metrics, metrics_values, saved_experiment_parameters = compute_metrics(args_params)
                parameters_keys, parameters_values,_ = extract_experiment_parameters(saved_experiment_parameters)
                if not is_header_printed:
                    with open(experiment_results_file_name, 'w') as f:
                        f.write('experiment_dir_name;' + parameters_keys + saved_metrics + '\n')
                    is_header_printed = True

                with open(experiment_results_file_name, 'a') as f:
                    f.write(dir_name + ';' + parameters_values + metrics_values + '\n')

            except Exception as e:
                print('Error computing experiment dir:', args_params.experiment_dir)
                print(e)
                traceback.print_exc()

        print("\nCSVs for all experiments metrics results saved in:\n", experiment_results_file_name)
    else:
        compute_metrics(args_params)


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


def compute_js(ori_data, generated_data_sample):
    return JSdistanceMultivariate(ori_data, generated_data_sample)


def compute_metrics(args_params):
    metrics_list, path_to_save_metrics, saved_experiments_parameters, saved_metrics, dataset_info, path_to_save_sdv_figures = initialization(
        args_params)

    if (args_params.ori_data_filename):
        ori_data = np.loadtxt(args_params.ori_data_filename, delimiter=",", skiprows=1)
        ori_data_df = pd.DataFrame(ori_data, columns=dataset_info['column_config'])
    else:
        ori_data_df = loadtraces.get_alibaba_2018_trace(stride_seconds = dataset_info['timestamp_frequency_secs'])
        ori_data = ori_data_df.to_numpy()

    if "tsne" in metrics_list or "pca" in metrics_list:
        generate_visualization_figures(args_params, path_to_save_metrics, metrics_list, ori_data)
        metrics_list.remove("tsne")
        metrics_list.remove("pca")

    metrics_results = {}
    avg_results = {}
    for metric in metrics_list:
        if metric != 'sdv-diagnostic':
            metrics_results[metric] = []
        if metric == 'mmd' or metric == 'dtw' or metric == 'kl' or metric == 'hi' or metric == 'ks' or metric == 'js':
            for column in range(ori_data.shape[1]):
                metrics_results[metric + '-' + str(column)] = []
        if metric == 'sdv-quality':
            metrics_results[metric + '-column_shapes'] = []
            metrics_results[metric + '-column_pair_trends'] = []
        if metric == 'sdv-diagnostic':
            metrics_results[metric + '-synthesis'] = []
            metrics_results[metric + '-coverage'] = []
            metrics_results[metric + '-boundaries'] = []

        n_files_iteration = 0
        total_files = len(fnmatch.filter(os.listdir(args_params.experiment_dir + '/generated_data'), '*.csv'))
        progress_bar2 = tqdm(os.listdir(args_params.experiment_dir + '/generated_data'), colour='blue', leave=False)
        for filename in progress_bar2:
            progress_bar2.set_description(f'Computing {metric:20} [{n_files_iteration+1}/{total_files}]')
            f = os.path.join(args_params.experiment_dir + '/generated_data', filename)
            if os.path.isfile(f):  # checking if it is a file
                generated_data_sample = np.loadtxt(f, delimiter=",")
                seq_len = len(generated_data_sample)
                ori_data_sample = get_ori_data_sample(seq_len, ori_data)
                computed_metric = 0
                if metric == 'mmd':  # mayor valor más distintas son
                    computed_metric = mmd_rbf(X=ori_data_sample, Y=generated_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            mmd_rbf(generated_data_sample[:, column].reshape(-1, 1),
                                    ori_data_sample[:, column].reshape(-1, 1)))
                if metric == 'dtw':  # mayor valor más distintas son
                    computed_metric = compute_dtw(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            compute_dtw(generated_data_sample[:, column].reshape(-1, 1),
                                        ori_data_sample[:, column].reshape(-1, 1)))
                if metric == 'kl':  # mayor valor peor
                    computed_metric = KLdivergence(ori_data, generated_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            KLDivergenceUnivariate(ori_data_sample[:, column].reshape(-1, 1),
                                                   generated_data_sample[:, column].reshape(-1, 1))[0])

                if metric == 'js':
                    computed_metric = compute_js(ori_data, generated_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            JSdistance(ori_data_sample[:, column].reshape(-1, 1),
                                       generated_data_sample[:, column].reshape(-1, 1)))
                if metric == 'ks':  # menor valor mejor
                    computed_metric = compute_ks(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            compute_ks(generated_data_sample[:, column].reshape(-1, 1),
                                       ori_data_sample[:, column].reshape(-1, 1)))
                if metric == 'cc':  # mayor valor peor. covarianza
                    computed_metric = compute_cc(generated_data_sample, ori_data_sample)
                if metric == 'cp':  # mayor valor peor. coeficiente de pearson
                    computed_metric = compute_cp(generated_data_sample, ori_data)
                if metric == 'hi':  # mayor valor peor
                    computed_metric = compute_hi(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            compute_hi(generated_data_sample[:, column].reshape(-1, 1),
                                       ori_data_sample[:, column].reshape(-1, 1)))
                if metric == 'sdv-quality':
                    if n_files_iteration % args_params.stride == 0:
                        computed_metric, column_shapes, column_pair_trends = compute_sdv_quality_metrics(dataset_info,
                                                                                                         generated_data_sample,
                                                                                                         n_files_iteration,
                                                                                                         ori_data_df,
                                                                                                         path_to_save_sdv_figures)
                if metric == 'sdv-diagnostic':
                    if n_files_iteration % args_params.stride == 0:
                        diagnostic_synthesis, diagnostic_coverage, diagnostic_boundaries = compute_sdv_diagnostic_metrics(
                            dataset_info, generated_data_sample,
                            n_files_iteration, ori_data_df,
                            path_to_save_sdv_figures)
                if metric == 'evolution_figures':
                    if n_files_iteration % args_params.stride == 0:  # generates a 10% of the figures
                        create_usage_evolution(generated_data_sample, ori_data, ori_data_sample,
                                               path_to_save_metrics + 'figures/', n_files_iteration, dataset_info)
                if metric != 'evolution_figures':
                    if metric != 'sdv-diagnostic':
                        if metric != 'sdv-quality':
                            metrics_results[metric].append(computed_metric)
                        elif metric == 'sdv-quality' and n_files_iteration % args_params.stride == 0:
                            metrics_results[metric].append(computed_metric)
                    if metric == 'sdv-quality' and n_files_iteration % args_params.stride == 0:
                        metrics_results[metric + '-column_shapes'].append(column_shapes)
                        metrics_results[metric + '-column_pair_trends'].append(column_pair_trends)
                    if metric == 'sdv-diagnostic' and n_files_iteration % args_params.stride == 0:
                        metrics_results[metric + '-synthesis'].append(diagnostic_synthesis)
                        metrics_results[metric + '-coverage'].append(diagnostic_coverage)
                        metrics_results[metric + '-boundaries'].append(diagnostic_boundaries)

                n_files_iteration += 1
        #print('')

    for metric, results in metrics_results.items():
        if metric != 'tsne' and metric != 'pca' and metric != 'evolution_figures':
            avg_results[metric] = statistics.mean(metrics_results[metric])
    saved_metrics, metrics_values, = save_metrics(avg_results, metrics_results, path_to_save_metrics,
                                                  saved_experiments_parameters, saved_metrics)
    return saved_metrics, metrics_values, saved_experiments_parameters


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
    
def compute_sdv_quality_metrics(dataset_info, generated_data_sample, n_files_iteration, ori_data_df,
                                path_to_save_sdv_quality_figures):
    report = QualityReport()
    generated_data_sample_df = pd.DataFrame(generated_data_sample,
                                            columns=dataset_info['column_config'])
    print("\nBefore no printing")
    with HiddenPrints():
        print("SHOULD NOT BE PRINTED")
        report.generate(ori_data_df, generated_data_sample_df, dataset_info['metadata'])

    print("After no printing")

    fig_column_shapes = report.get_visualization(property_name='Column Shapes')
    fig_column_pair_trends = report.get_visualization(property_name='Column Pair Trends')
    fig_column_shapes.write_image(
        path_to_save_sdv_quality_figures + '/column_shapes-' + str(n_files_iteration) + '.pdf')
    fig_column_pair_trends.write_image(
        path_to_save_sdv_quality_figures + '/column_pair_trends-' + str(n_files_iteration) + '.pdf')

    return report.get_score(), report.get_properties().iloc[0, 1], report.get_properties().iloc[1, 1]


def compute_sdv_diagnostic_metrics(dataset_info, generated_data_sample, n_files_iteration, ori_data_df,
                                   path_to_save_sdv_quality_figures):
    report = DiagnosticReport()
    generated_data_sample_df = pd.DataFrame(generated_data_sample,
                                            columns=dataset_info['column_config'])
    report.generate(ori_data_df, generated_data_sample_df, dataset_info['metadata'])
    fig_synthesis = report.get_visualization(property_name='Synthesis')
    fig_coverage = report.get_visualization(property_name='Coverage')
    fig_boundaries = report.get_visualization(property_name='Boundaries')

    fig_synthesis.write_image(
        path_to_save_sdv_quality_figures + '/synthesis-' + str(n_files_iteration) + '.pdf')
    fig_coverage.write_image(
        path_to_save_sdv_quality_figures + '/coverage-' + str(n_files_iteration) + '.pdf')
    fig_boundaries.write_image(
        path_to_save_sdv_quality_figures + '/boundaries-' + str(n_files_iteration) + '.pdf')

    return report.get_properties()['Synthesis'], report.get_properties()['Coverage'], report.get_properties()[
        'Boundaries']


def initialization(args_params):
    path_to_save_metrics = args_params.experiment_dir + "/evaluation_metrics/"
    path_to_save_sdv_figures = path_to_save_metrics + 'figures/sdv/'
    if args_params.recursive == 'true' and os.path.isfile(args_params.experiment_dir + '/../parameters.txt'):
        parameters_file = open(args_params.experiment_dir + '/../parameters.txt', 'r')
    else:
        parameters_file = open(args_params.experiment_dir + '/parameters.txt', 'r')
    saved_metrics ="no metrics"
    if args_params.recursive == 'true' and os.path.isfile(args_params.experiment_dir + '/../metrics.txt'):
        metrics_file = open(args_params.experiment_dir + '/../metrics.txt', 'r')
        saved_metrics = metrics_file.readline()
    elif os.path.isfile(args_params.experiment_dir + '/metrics.txt'):
        metrics_file = open(args_params.experiment_dir + '/metrics.txt', 'r')
        saved_metrics = metrics_file.readline()

    saved_experiments_parameters = parameters_file.readline()
    os.makedirs(path_to_save_metrics, exist_ok=True)
    os.makedirs(path_to_save_metrics + '/figures/', exist_ok=True)
    os.makedirs(path_to_save_sdv_figures, exist_ok=True)

    if args_params.metrics == 'all':
        args_params.metrics = 'mmd,dtw,js,kl,ks,cc,cp,hi,sdv-quality,sdv-diagnostic,evolution_figures'

    metrics_list = [metric for metric in args_params.metrics.split(',')]

    if args_params.trace == 'alibaba2018':
        dataset_info = {
            "timestamp_frequency_secs": 300,
            "column_config": {
                "cpu_util_percent": {
                    "column_index": 0,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "mem_util_percent": {
                    "column_index": 1,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "net_in": {
                    "column_index": 2,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "net_out": {
                    "column_index": 3,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "disk_io_percent": {
                    "column_index": 4,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                }

            },
            "metadata": {
                "fields": {
                    "cpu_util_percent": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "mem_util_percent": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "net_in": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "net_out": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "disk_io_percent": {
                        "type": "numerical",
                        "subtype": "float"
                    }
                }
            }
        }
    elif args_params.trace == 'google2019':
        dataset_info = {
            "timestamp_frequency_secs": 300,
            "column_config": {
                "cpu": {
                    "column_index": 0,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "mem": {
                    "column_index": 1,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "assigned_mem": {
                    "column_index": 2,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "cycles_per_instruction": {
                    "column_index": 3
                }
            },
            "metadata": {
                "fields": {
                    "cpu": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "mem": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "assigned_mem": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "cycles_per_instruction": {
                        "type": "numerical",
                        "subtype": "float"
                    }
                }
            }
        }
    elif args_params.trace == 'azure_v2':
        dataset_info = {
            "timestamp_frequency_secs": 300,
            "column_config": {
                "cpu_total": {
                    "column_index": 0
                },
                "mem_total": {
                    "column_index": 1
                }
            },
            "metadata": {
                "fields": {
                    "cpu_total": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "mem_total": {
                        "type": "numerical",
                        "subtype": "float"
                    }
                }
            }
        }
    elif args_params.trace == 'reddit':
        dataset_info = {
            "timestamp_frequency_secs": 3600,
            "column_config": {
                "interactions": {
                    "column_index": 0
                }
            }
        }

    return metrics_list, path_to_save_metrics, saved_experiments_parameters, saved_metrics, dataset_info, path_to_save_sdv_figures


def preprocess_dataset(ori_data, seq_len):
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


def results_for_excel(avg_results):
    metrics_values = ''
    computed_metrics = ''
    for metric_name in avg_results:
        computed_metrics += metric_name + ';'
        metrics_values += str(avg_results[metric_name]).replace('.', ',') + ';'

    return computed_metrics, metrics_values


def save_metrics(avg_results, metrics_results, path_to_save_metrics, saved_experiments_parameters, saved_metrics):
    _, _, parameters_dict = extract_experiment_parameters(saved_experiments_parameters)
    if 'data_name' in parameters_dict:
        data_name = parameters_dict['data_name']
    elif 'trace' in parameters_dict:
        data_name = parameters_dict['trace']
    else:
        data_name = 'not_found'
    if 'iteration' in parameters_dict:
        iteration = parameters_dict['iteration']
    elif 'epochs' in parameters_dict:
        iteration = parameters_dict['epochs']
    else:
        iteration = 'not_found'

    seq_len = parameters_dict['seq_len']
    with open(
            path_to_save_metrics + '/metrics-' + data_name + '-iterations-' + iteration + '-seq_len' + seq_len + '.txt',
            'w') as f:
        f.write(saved_experiments_parameters + '\n\n')
        f.write(saved_metrics + '\n\n')
        f.write(repr(avg_results) + '\n')
        computed_metrics, metrics_values = results_for_excel(avg_results)
        f.write(
            'Results of the following metrics: ' + computed_metrics + ' in spanish locale Excel format:' + '\n' + metrics_values + '\n')
        f.write(repr(metrics_results))
    #print("Metrics saved in file", f.name)
    return computed_metrics, metrics_values


def compute_ks(generated_data_sample, ori_data_sample):
    column_indexes = range(generated_data_sample.shape[1])
    return statistics.mean(
        [scipy.stats.ks_2samp(generated_data_sample[:, column_index], ori_data_sample[:, column_index])[0] for
         column_index in column_indexes])


def compute_dtw(generated_data_sample, ori_data_sample):
    sample_length = len(generated_data_sample)
    processed_generated_data_sample = np.insert(generated_data_sample, 0, np.ones(sample_length, dtype=int), axis=1)
    processed_generated_data_sample = np.insert(processed_generated_data_sample, 0, range(sample_length), axis=1)
    processed_ori_data_sample = np.insert(ori_data_sample, 0, np.ones(sample_length, dtype=int), axis=1)
    processed_ori_data_sample = np.insert(processed_ori_data_sample, 0, range(sample_length), axis=1)

    return dtw_ndim.distance_fast(processed_generated_data_sample, processed_ori_data_sample)


def compute_cp(generated_data_sample, ori_data_sample):
    # normalized_ori_data_sample = normalize_start_time_to_zero(ori_data_sample)
    # normalized_generated_data_sample = normalize_start_time_to_zero(generated_data_sample)
    ori_data_sample_pearson = np.corrcoef(ori_data_sample, rowvar=False)
    generated_data_sample_pearson = np.corrcoef(generated_data_sample, rowvar=False)
    correlation_diff_matrix = ori_data_sample_pearson - generated_data_sample_pearson
    l1_norms_avg = np.mean([np.linalg.norm(row) for row in correlation_diff_matrix])
    return l1_norms_avg


def compute_cc(generated_data_sample, ori_data_sample):
    # normalized_ori_data_sample = normalize_start_time_to_zero(ori_data_sample)
    # normalized_generated_data_sample = normalize_start_time_to_zero(generated_data_sample)
    ori_data_sample_covariance = np.cov(ori_data_sample)
    generated_data_covariance = np.cov(generated_data_sample)
    covariance_diff_matrix = ori_data_sample_covariance - generated_data_covariance
    l1_norms_avg = np.mean([np.linalg.norm(row) for row in covariance_diff_matrix])
    return l1_norms_avg


def compute_hi(generated_data_sample, ori_data):
    # normalized_ori_data_sample = normalize_start_time_to_zero(ori_data)
    # normalized_generated_data_sample = normalize_start_time_to_zero(generated_data_sample)
    histogram_diff_matrix = []
    for column in range(0, ori_data.shape[1]):
        ori_data_column_values = ori_data[:, column]
        ori_histogram, ori_bin_edges = np.histogram(ori_data_column_values)
        generated_data_column_values = generated_data_sample[:, column]
        generated_histogram, generated_bin_edges = np.histogram(generated_data_column_values)
        column_histogram_diff = ori_histogram - generated_histogram
        histogram_diff_matrix.append(column_histogram_diff)
    histogram_diff_matrix = np.asmatrix(histogram_diff_matrix)
    l1_norms_histogram_diff = np.apply_along_axis(np.linalg.norm, 1, histogram_diff_matrix)

    l1_norms_histogram_diff_avg = l1_norms_histogram_diff.mean()

    return l1_norms_histogram_diff_avg


def normalize_start_time_to_zero(sample):
    timestamp_column = sample[:, 0]
    min_timestamp = np.min(timestamp_column)
    normalized_timestamp_column = timestamp_column - min_timestamp
    sample[:, 0] = normalized_timestamp_column
    return sample


def generate_visualization_figures(args_param, directory_name, metrics_list, ori_data):
    generated_data = []
    n_samples = 0
    seq_len = 0
    for filename in os.listdir(args_param.experiment_dir + '/generated_data'):
        f = os.path.join(args_param.experiment_dir + '/generated_data', filename)
        if os.path.isfile(f):  # checking if it is a file
            n_samples = n_samples + 1
            generated_data_sample = np.loadtxt(f, delimiter=",")
            seq_len = len(generated_data_sample)
            generated_data.append(generated_data_sample)

    ori_data_for_visualization = preprocess_dataset(ori_data, seq_len)
    if "tsne" in metrics_list:
        visualization(ori_data=ori_data_for_visualization, generated_data=generated_data, analysis='tsne',
                      n_samples=n_samples, path_for_saving_images=directory_name)
    if "pca" in metrics_list:
        visualization(ori_data=ori_data_for_visualization, generated_data=generated_data, analysis='pca',
                      n_samples=n_samples, path_for_saving_images=directory_name)


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ori_data_filename',
        default='data/mu_day3_cut.csv',
        type=str)
    parser.add_argument(
        '--experiment_dir',
        type=str)
    parser.add_argument(
        '--metrics',
        default='mmd',
        type=str)
    parser.add_argument(
        '--trace',
        default='alibaba2018',
        type=str)
    parser.add_argument(
        '--recursive',
        default='false',
        type=str)
    parser.add_argument(
        '--stride',
        default='1',
        type=int)

    args = parser.parse_args()
    main(args)
