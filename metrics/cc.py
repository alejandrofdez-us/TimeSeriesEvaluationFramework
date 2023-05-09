import numpy as np


def cc(generated_data_sample, ori_data_sample):
    # TODO: remove the following two commented lines?
    # normalized_ori_data_sample = normalize_start_time_to_zero(ori_data_sample)
    # normalized_generated_data_sample = normalize_start_time_to_zero(generated_data_sample)
    ori_data_sample_covariance = np.cov(ori_data_sample)
    generated_data_covariance = np.cov(generated_data_sample)
    covariance_diff_matrix = ori_data_sample_covariance - generated_data_covariance
    l1_norms_avg = np.mean([np.linalg.norm(row) for row in covariance_diff_matrix])
    return l1_norms_avg
