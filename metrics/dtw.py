import numpy as np
from dtaidistance import dtw_ndim

def dtw(generated_data_sample, ori_data_sample):
    metric_result = f"Multivariate: {compute_dtw(ori_data_sample, generated_data_sample)}"

    for column in range(generated_data_sample.shape[1]):
        metric_result = metric_result + f" Column {column}: {compute_dtw(generated_data_sample[:, column].reshape(-1, 1), ori_data_sample[:, column].reshape(-1, 1))}"

    return metric_result

def compute_dtw(generated_data_sample, ori_data_sample):
    sample_length = len(generated_data_sample)
    processed_generated_data_sample = np.insert(generated_data_sample, 0, np.ones(sample_length, dtype=int), axis=1)
    processed_generated_data_sample = np.insert(processed_generated_data_sample, 0, range(sample_length), axis=1)
    processed_ori_data_sample = np.insert(ori_data_sample, 0, np.ones(sample_length, dtype=int), axis=1)
    processed_ori_data_sample = np.insert(processed_ori_data_sample, 0, range(sample_length), axis=1)

    return dtw_ndim.distance_fast(processed_generated_data_sample, processed_ori_data_sample)
