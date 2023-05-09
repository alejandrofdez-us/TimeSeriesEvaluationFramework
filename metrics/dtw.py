import numpy as np
from dtaidistance import dtw_ndim

def compute_dtw(generated_data_sample, ori_data_sample):
    sample_length = len(generated_data_sample)
    processed_generated_data_sample = np.insert(generated_data_sample, 0, np.ones(sample_length, dtype=int), axis=1)
    processed_generated_data_sample = np.insert(processed_generated_data_sample, 0, range(sample_length), axis=1)
    processed_ori_data_sample = np.insert(ori_data_sample, 0, np.ones(sample_length, dtype=int), axis=1)
    processed_ori_data_sample = np.insert(processed_ori_data_sample, 0, range(sample_length), axis=1)

    return dtw_ndim.distance_fast(processed_generated_data_sample, processed_ori_data_sample)
