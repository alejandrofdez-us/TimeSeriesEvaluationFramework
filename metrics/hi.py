import numpy as np


def hi(generated_data_sample, ori_data):
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