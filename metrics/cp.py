import numpy as np


def cp(generated_data_sample, ori_data_sample):
    # TODO: remove the following two commented lines?
    # normalized_ori_data_sample = normalize_start_time_to_zero(ori_data_sample)
    # normalized_generated_data_sample = normalize_start_time_to_zero(generated_data_sample)
    ori_data_sample_pearson = np.corrcoef(ori_data_sample, rowvar=False)
    generated_data_sample_pearson = np.corrcoef(generated_data_sample, rowvar=False)
    correlation_diff_matrix = ori_data_sample_pearson - generated_data_sample_pearson
    l1_norms_avg = np.mean([np.linalg.norm(row) for row in correlation_diff_matrix])
    return l1_norms_avg
