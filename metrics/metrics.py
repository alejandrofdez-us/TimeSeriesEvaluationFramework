import statistics

import numpy as np
import scipy
from dtaidistance import dtw_ndim
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport

from metrics.kl import JSdistanceMultivariate

from tqdm import tqdm
from functools import partialmethod


def compute_sdv_quality_metrics(dataset_info, generated_data_sample_df, n_files_iteration, ori_data_df,
                                path_to_save_sdv_quality_figures):
    # silence the tqdm
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    report = QualityReport()
    report.generate(ori_data_df, generated_data_sample_df, dataset_info['metadata'])
    fig_column_shapes = report.get_visualization(property_name='Column Shapes')
    fig_column_pair_trends = report.get_visualization(property_name='Column Pair Trends')
    fig_column_shapes.write_image(
        path_to_save_sdv_quality_figures + '/column_shapes-' + str(n_files_iteration) + '.pdf')
    fig_column_pair_trends.write_image(
        path_to_save_sdv_quality_figures + '/column_pair_trends-' + str(n_files_iteration) + '.pdf')

    return report.get_score(), report.get_properties().iloc[0, 1], report.get_properties().iloc[1, 1]


def compute_sdv_diagnostic_metrics(dataset_info, generated_data_sample_df, n_files_iteration, ori_data_df,
                                   path_to_save_sdv_quality_figures):
    report = DiagnosticReport()
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


def compute_js(ori_data, generated_data_sample):
    return JSdistanceMultivariate(ori_data, generated_data_sample)
