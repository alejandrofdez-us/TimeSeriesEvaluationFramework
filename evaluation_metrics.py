import argparse
import copy
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
from natsort import natsorted
from tqdm import tqdm
import multiprocessing
from pathlib import Path



from evolution_figures import create_usage_evolution, generate_inter_experiment_figures, generate_tsne_pca_figures
from metrics.kl import KLDivergenceUnivariate
from metrics.kl import KLdivergence, JSdistance
from metrics.metrics import compute_sdv_quality_metrics, compute_sdv_diagnostic_metrics, compute_ks, compute_dtw, \
    compute_cp, compute_cc, compute_hi, compute_js
from metrics.mmd import mmd_rbf
from utils import split_ori_data_strided, get_most_similar_ori_data_sample, \
    extract_experiment_parameters, save_metrics, print_csv_result_row, print_csv_header, \
    print_previously_computed_experiments_metrics

from concurrent.futures import ProcessPoolExecutor

MAX_WORKERS = int(ProcessPoolExecutor()._max_workers/2)
CHUNK_SIZE = 1




def main(args_params):
    if args_params.recursive:
        root_dir = args_params.experiment_dir + '/'
        experiment_results_file_name = f'{root_dir}experiments_metrics-{os.path.basename(os.path.normpath(root_dir))}-{datetime.now().strftime("%j-%H-%M-%S")}.csv'
        experiment_directories_to_be_computed = []
        experiment_directories_previously_computed = []
        for subdir, dirs, files in os.walk(root_dir):
            if 'generated_data' in dirs:
                if args_params.recompute_metrics:
                    experiment_directories_to_be_computed.append(subdir)
                else:
                    if not 'evaluation_metrics' in dirs:
                        experiment_directories_to_be_computed.append(subdir)
                    elif 'evaluation_metrics' in dirs:
                        experiment_directories_previously_computed.append(subdir)
        if (experiment_directories_previously_computed):
            print(f'Found previous metrics computations for {len(experiment_directories_previously_computed)} experiments. Skipping.')

        experiment_directories_to_be_computed = natsorted(experiment_directories_to_be_computed)
        experiment_directories_previously_computed = natsorted(experiment_directories_previously_computed)
        if args.metrics:
            is_header_printed = False
            args_params_array = []
            for dir_name in experiment_directories_to_be_computed:
                args_params.experiment_dir = dir_name
                args_params_array.append(copy.deepcopy(args_params))

            try:
                with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
                    results_progress_bar = tqdm(
                        pool.imap_unordered(compute_metrics, args_params_array, chunksize=CHUNK_SIZE),
                        total=len(args_params_array),
                        desc='Computing metrics'
                    )
                    for saved_metrics, metrics_values, saved_experiment_parameters, experiment_dir_name in results_progress_bar:
                        parameters_keys, parameters_values, _ = extract_experiment_parameters(
                            saved_experiment_parameters)
                        if not is_header_printed:
                            print_csv_header(experiment_results_file_name, parameters_keys, saved_metrics)
                            print_previously_computed_experiments_metrics(experiment_directories_previously_computed,experiment_results_file_name)
                            is_header_printed = True

                        print_csv_result_row(experiment_dir_name, experiment_results_file_name, metrics_values,
                                             parameters_values)
                        results_progress_bar.set_description(f'Saved metrics of {Path(*Path(experiment_dir_name).parts[-3:])}')
            except Exception as e:
                print('Error computing experiment dir:', args_params.experiment_dir)
                print(e)
                traceback.print_exc()

            print("\nCSVs for all experiments metrics results saved in:\n", experiment_results_file_name)
        if args_params.inter_experiment_figures and experiment_directories_to_be_computed:
            generate_inter_experiment_figures(root_dir, experiment_directories_to_be_computed, args_params)
    else:
        compute_metrics(args_params)


def compute_metrics(args_params):
    metrics_list, path_to_save_metrics, saved_experiments_parameters, saved_metrics, dataset_info, ori_data, ori_data_df = initialization(
        args_params)

    _, _, parameters_dict = extract_experiment_parameters(saved_experiments_parameters)

    if "tsne" in metrics_list or "pca" in metrics_list:
        generate_tsne_pca_figures(args_params, f'{path_to_save_metrics}/figures/', metrics_list, ori_data)
        metrics_list.remove("tsne")
        metrics_list.remove("pca")

    avg_results = {}
    metrics_results = initializa_metrics_results_structure(metrics_list, ori_data)

    sorted_generated_samples_dict = load_dtw_sorted_samples_objects(args_params, dataset_info, ori_data_df)
    n_files_iteration = 0
    for sample_filename, sample_objects in sorted_generated_samples_dict.items():
        generated_data_sample = sample_objects['generated_data_sample']
        generated_data_sample_df = sample_objects['generated_data_sample_df']
        ori_data_sample = sample_objects['ori_data_sample']

        computed_metric = 0
        metric_iteration = 0
        for metric in metrics_list:
            if metric == 'mmd':
                computed_metric = mmd_rbf(X=ori_data_sample, Y=generated_data_sample)
                for column in range(generated_data_sample.shape[1]):
                    metrics_results[metric + '-' + str(column)].append(
                        mmd_rbf(generated_data_sample[:, column].reshape(-1, 1),
                                ori_data_sample[:, column].reshape(-1, 1)))
            if metric == 'dtw':
                computed_metric = sample_objects['computed_dtw']
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
            if n_files_iteration % args_params.stride_metrics == 0:
                if metric == 'sdv-quality':
                    computed_metric, column_shapes, column_pair_trends = compute_sdv_quality_metrics(dataset_info,
                                                                                                     generated_data_sample_df,
                                                                                                     sample_filename,
                                                                                                     ori_data_df,
                                                                                                     f'{path_to_save_metrics}figures/sdv')
                if metric == 'sdv-diagnostic':
                    diagnostic_synthesis, diagnostic_coverage, diagnostic_boundaries = compute_sdv_diagnostic_metrics(
                            dataset_info, generated_data_sample_df,
                            sample_filename, ori_data_df,
                            f'{path_to_save_metrics}figures/sdv')
                if metric == 'evolution_figures' and args_params.only_best_samples_figures > n_files_iteration:
                    create_usage_evolution(generated_data_sample, generated_data_sample_df, ori_data, ori_data_sample,
                                               path_to_save_metrics + 'figures/',
                                               f'rank-{n_files_iteration}-{sample_filename}', dataset_info, args_params.generate_deltas)

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
    return saved_metrics, metrics_values, saved_experiments_parameters, args_params.experiment_dir


def load_dtw_sorted_samples_objects(args_params, dataset_info, ori_data_df):
    sorted_sample_names = natsorted(fnmatch.filter(os.listdir(args_params.experiment_dir + '/generated_data'), '*.csv'))
    generated_samples_dict = {}

    ori_data_windows_numpy = None
    for filename in sorted_sample_names:
        f = os.path.join(args_params.experiment_dir + '/generated_data', filename)
        if os.path.isfile(f):
            generated_data_sample = np.loadtxt(f, delimiter=",")
            generated_data_sample_df = pd.DataFrame(generated_data_sample,
                                                    columns=dataset_info['column_config'])

            if ori_data_windows_numpy is None:
                ori_data_windows_numpy = split_ori_data_strided(ori_data_df, generated_data_sample.shape[0],
                                                                args_params.stride_ori_data_windows)

            ori_data_sample, computed_dtw = get_most_similar_ori_data_sample(ori_data_windows_numpy,
                                                                             generated_data_sample)
            generated_samples_dict[filename] = {}
            generated_samples_dict[filename]['generated_data_sample'] = generated_data_sample
            generated_samples_dict[filename]['generated_data_sample_df'] = generated_data_sample_df
            generated_samples_dict[filename]['ori_data_sample'] = ori_data_sample
            generated_samples_dict[filename]['computed_dtw'] = computed_dtw
    generated_samples_dict = {k: v for k, v in
                              sorted(generated_samples_dict.items(), key=lambda k_v: k_v[1]['computed_dtw'])}
    return generated_samples_dict


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
    elif args_params.recursive and os.path.isfile(args_params.experiment_dir + '/../../parameters.txt'):
        parameters_file = open(args_params.experiment_dir + '/../../parameters.txt', 'r')
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
        args_params.metrics = 'js,mmd,dtw,kl,ks,cc,cp,hi,evolution_figures,tsne,pca,sdv-quality,sdv-diagnostic'
    elif args_params.metrics == 'all-no-sdv':
        args_params.metrics = 'js,mmd,dtw,kl,ks,cc,cp,hi,evolution_figures,tsne,pca'
    elif args_params.metrics == 'all-no-sdv-no-tsne-pca':
        args_params.metrics = 'js,mmd,dtw,kl,ks,cc,cp,hi,evolution_figures'

    metrics_list = [metric for metric in args_params.metrics.split(',')]

    dataset_info = loadtraces.get_dataset_info(trace_name=args_params.trace, trace_type=args_params.trace_type,
                                               stride_seconds=args_params.trace_timestep)

    if args_params.ori_data_filename:
        ori_data = np.loadtxt(args_params.ori_data_filename, delimiter=",", skiprows=1)
        ori_data_df = pd.DataFrame(ori_data, columns=dataset_info['column_config'])
    else:
        ori_data_df = loadtraces.get_trace(args_params.trace, trace_type=args_params.trace_type,
                                           stride_seconds=args_params.trace_timestep)
        ori_data = ori_data_df.to_numpy()

    return metrics_list, path_to_save_metrics, saved_experiments_parameters, previously_saved_metrics, dataset_info, ori_data, ori_data_df


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
        choices=['azure_v2', 'google2019', 'alibaba2018'],
        type=str)
    parser.add_argument(
        '--recursive',
        default=False,
        type=lambda x: bool(distutils.util.strtobool(str(x))))
    parser.add_argument(
        '--generate_deltas',
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
        '--trace_timestep',
        default='300',
        type=int)
    parser.add_argument(
        '--trace_type',
        default='machine_usage',
        type=str)
    parser.add_argument(
        '--inter_experiment_figures',
        default=False,
        type=lambda x: bool(distutils.util.strtobool(str(x))))
    parser.add_argument(
        '--recompute_metrics',
        default=False,
        type=lambda x: bool(distutils.util.strtobool(str(x))))
    parser.add_argument(
        '--only_best_samples_figures',
        default='10',
        type=int)

    args = parser.parse_args()
    main(args)
