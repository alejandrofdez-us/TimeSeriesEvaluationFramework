import argparse
import distutils
from distutils import util
import fnmatch
import os
import statistics
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from datacentertracesdatasets import loadtraces
from tqdm import tqdm

from evolution_figures import create_usage_evolution, generate_inter_experiment_figures
from metrics.kl import KLDivergenceUnivariate
from metrics.kl import KLdivergence, JSdistance
from metrics.metrics import compute_sdv_quality_metrics, compute_sdv_diagnostic_metrics, compute_ks, compute_dtw, \
    compute_cp, compute_cc, compute_hi, compute_js
from metrics.mmd import mmd_rbf
from utils import get_dataset_info, split_ori_data_strided, get_most_similar_ori_data_sample, \
    extract_experiment_parameters, save_metrics


def main(args_params):
    if args_params.recursive:
        root_dir = args_params.experiment_dir + '/'
        experiment_results_file_name = root_dir + 'experiments_metrics-' + datetime.now().strftime(
            "%j-%H-%M-%S") + '.csv'
        experiment_directories = []
        for subdir, dirs, files in os.walk(root_dir):
            if 'generated_data' in dirs:
                experiment_directories.append(subdir)

        if args.metrics:
            is_header_printed = False
            progress_bar_general = tqdm(experiment_directories, colour="red", position=0)
            for dir_name in progress_bar_general:
                args_params.experiment_dir = dir_name
                try:
                    progress_bar_general.set_description(
                        "Computing metrics for directory " + os.path.basename(os.path.normpath(dir_name)))
                    # print("Computing metrics for directory ", dir_name)
                    saved_metrics, metrics_values, saved_experiment_parameters = compute_metrics(args_params)
                    parameters_keys, parameters_values, _ = extract_experiment_parameters(saved_experiment_parameters)
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
        if args_params.inter_experiment_figures:
            generate_inter_experiment_figures(root_dir, experiment_directories, args_params.trace)
    else:
        compute_metrics(args_params)


def compute_metrics(args_params):
    metrics_list, path_to_save_metrics, saved_experiments_parameters, saved_metrics, dataset_info, path_to_save_sdv_figures, ori_data, ori_data_df = initialization(
        args_params)

    _, _, parameters_dict = extract_experiment_parameters(saved_experiments_parameters)
    ori_data_windows_numpy = split_ori_data_strided(ori_data_df, int(parameters_dict['seq_len']),
                                                    args_params.stride_ori_data_windows)
    avg_results = {}
    metrics_results = initializa_metrics_results_structure(metrics_list, ori_data)

    n_files_iteration = 0
    total_files = len(fnmatch.filter(os.listdir(args_params.experiment_dir + '/generated_data'), '*.csv'))
    sorted_sample_names = sorted(fnmatch.filter(os.listdir(args_params.experiment_dir + '/generated_data'), '*.csv'),
                                 key=lambda fileName: int(fileName.split('.')[0].split('_')[1]))
    progress_bar = tqdm(sorted_sample_names, colour='green', position=1)
    for filename in progress_bar:
        progress_bar.set_description(f'Computing {filename:10} [{n_files_iteration + 1}/{total_files}]')
        f = os.path.join(args_params.experiment_dir + '/generated_data', filename)
        if os.path.isfile(f):
            generated_data_sample = np.loadtxt(f, delimiter=",")
            generated_data_sample_df = pd.DataFrame(generated_data_sample,
                                                    columns=dataset_info['column_config'])
            ori_data_sample = get_most_similar_ori_data_sample(ori_data_windows_numpy, generated_data_sample)
            computed_metric = 0
            progress_bar2 = tqdm(metrics_list, colour='blue', position=2, leave=False)
            metric_iteration = 0
            for metric in progress_bar2:
                progress_bar2.set_description(f'Computing {metric:10} [{metric_iteration + 1}/{len(metrics_list)}]')
                if metric == 'mmd':
                    computed_metric = mmd_rbf(X=ori_data_sample, Y=generated_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            mmd_rbf(generated_data_sample[:, column].reshape(-1, 1),
                                    ori_data_sample[:, column].reshape(-1, 1)))
                if metric == 'dtw':
                    computed_metric = compute_dtw(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            compute_dtw(generated_data_sample[:, column].reshape(-1, 1),
                                        ori_data_sample[:, column].reshape(-1, 1)))
                if metric == 'kl':
                    computed_metric = KLdivergence(ori_data_sample, generated_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            KLDivergenceUnivariate(ori_data_sample[:, column].reshape(-1, 1),
                                                   generated_data_sample[:, column].reshape(-1, 1))[0])
                if metric == 'js':
                    computed_metric = compute_js(ori_data_sample, generated_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            JSdistance(ori_data_sample[:, column].reshape(-1, 1),
                                       generated_data_sample[:, column].reshape(-1, 1)))
                if metric == 'ks':
                    computed_metric = compute_ks(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            compute_ks(generated_data_sample[:, column].reshape(-1, 1),
                                       ori_data_sample[:, column].reshape(-1, 1)))
                if metric == 'cc':
                    computed_metric = compute_cc(generated_data_sample, ori_data_sample)
                if metric == 'cp':
                    computed_metric = compute_cp(generated_data_sample, ori_data_sample)
                if metric == 'hi':
                    computed_metric = compute_hi(generated_data_sample, ori_data_sample)
                    for column in range(generated_data_sample.shape[1]):
                        metrics_results[metric + '-' + str(column)].append(
                            compute_hi(generated_data_sample[:, column].reshape(-1, 1),
                                       ori_data_sample[:, column].reshape(-1, 1)))
                if metric == 'sdv-quality':
                    if n_files_iteration % args_params.stride_metrics == 0:
                        computed_metric, column_shapes, column_pair_trends = compute_sdv_quality_metrics(dataset_info,
                                                                                                         generated_data_sample_df,
                                                                                                         n_files_iteration,
                                                                                                         ori_data_df,
                                                                                                         path_to_save_sdv_figures)
                if metric == 'sdv-diagnostic':
                    if n_files_iteration % args_params.stride_metrics == 0:
                        diagnostic_synthesis, diagnostic_coverage, diagnostic_boundaries = compute_sdv_diagnostic_metrics(
                            dataset_info, generated_data_sample_df,
                            n_files_iteration, ori_data_df,
                            path_to_save_sdv_figures)
                if metric == 'evolution_figures':
                    if n_files_iteration % args_params.stride_metrics == 0:
                        create_usage_evolution(generated_data_sample, ori_data, ori_data_sample,
                                               path_to_save_metrics + 'figures/', n_files_iteration, dataset_info)
                if metric != 'evolution_figures':
                    if metric != 'sdv-diagnostic':
                        if metric != 'sdv-quality':
                            metrics_results[metric].append(computed_metric)
                        elif metric == 'sdv-quality' and n_files_iteration % args_params.stride_metrics == 0:
                            metrics_results[metric].append(computed_metric)
                    if metric == 'sdv-quality' and n_files_iteration % args_params.stride_metrics == 0:
                        metrics_results[metric + '-column_shapes'].append(column_shapes)
                        metrics_results[metric + '-column_pair_trends'].append(column_pair_trends)
                    if metric == 'sdv-diagnostic' and n_files_iteration % args_params.stride_metrics == 0:
                        metrics_results[metric + '-synthesis'].append(diagnostic_synthesis)
                        metrics_results[metric + '-coverage'].append(diagnostic_coverage)
                        metrics_results[metric + '-boundaries'].append(diagnostic_boundaries)
                metric_iteration += 1

            n_files_iteration += 1

    for metric, results in metrics_results.items():
        if metric != 'evolution_figures':
            avg_results[metric] = statistics.mean(metrics_results[metric])
    saved_metrics, metrics_values, = save_metrics(avg_results, metrics_results, path_to_save_metrics,
                                                  saved_experiments_parameters, saved_metrics)
    return saved_metrics, metrics_values, saved_experiments_parameters


def initializa_metrics_results_structure(metrics_list, ori_data):
    metrics_results = {}
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
    return metrics_results


def initialization(args_params):
    path_to_save_metrics = args_params.experiment_dir + "/evaluation_metrics/"
    path_to_save_sdv_figures = path_to_save_metrics + 'figures/sdv/'
    if args_params.recursive and os.path.isfile(args_params.experiment_dir + '/../parameters.txt'):
        parameters_file = open(args_params.experiment_dir + '/../parameters.txt', 'r')
    else:
        parameters_file = open(args_params.experiment_dir + '/parameters.txt', 'r')
    previously_saved_metrics = "no metrics"
    if args_params.recursive and os.path.isfile(args_params.experiment_dir + '/../metrics.txt'):
        metrics_file = open(args_params.experiment_dir + '/../metrics.txt', 'r')
        previously_saved_metrics = metrics_file.readline()
    elif os.path.isfile(args_params.experiment_dir + '/metrics.txt'):
        metrics_file = open(args_params.experiment_dir + '/metrics.txt', 'r')
        previously_saved_metrics = metrics_file.readline()

    saved_experiments_parameters = parameters_file.readline()
    os.makedirs(path_to_save_metrics, exist_ok=True)
    os.makedirs(path_to_save_metrics + '/figures/', exist_ok=True)
    os.makedirs(path_to_save_sdv_figures, exist_ok=True)

    if args_params.metrics == 'all':
        args_params.metrics = 'mmd,dtw,js,kl,ks,cc,cp,hi,sdv-quality,sdv-diagnostic,evolution_figures'

    metrics_list = [metric for metric in args_params.metrics.split(',')]

    dataset_info = get_dataset_info(args_params.trace)

    if (args_params.ori_data_filename):
        ori_data = np.loadtxt(args_params.ori_data_filename, delimiter=",", skiprows=1)
        ori_data_df = pd.DataFrame(ori_data, columns=dataset_info['column_config'])
    else:
        ori_data_df = loadtraces.get_alibaba_2018_trace(stride_seconds=dataset_info['timestamp_frequency_secs'])
        ori_data = ori_data_df.to_numpy()

    return metrics_list, path_to_save_metrics, saved_experiments_parameters, previously_saved_metrics, dataset_info, path_to_save_sdv_figures, ori_data, ori_data_df


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ori_data_filename',
        type=str)
    parser.add_argument(
        '--experiment_dir',
        type=str)
    parser.add_argument(
        '--metrics',
        type=str)
    parser.add_argument(
        '--trace',
        default='alibaba2018',
        type=str)
    parser.add_argument(
        '--recursive',
        default=False,
        type=lambda x: bool(distutils.util.strtobool(str(x))))
    parser.add_argument(
        '--stride_metrics',
        default='1',
        type=int)
    parser.add_argument(
        '--stride_ori_data_windows',
        default='1',
        type=int)
    parser.add_argument(
        '--inter_experiment_figures',
        default=False,
        type=lambda x: bool(distutils.util.strtobool(str(x))))

    args = parser.parse_args()
    main(args)
