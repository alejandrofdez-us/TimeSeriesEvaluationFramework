import scipy
import statistics


def ks(generated_data_sample, ori_data_sample):
    column_indexes = range(generated_data_sample.shape[1])
    return statistics.mean(
        [scipy.stats.ks_2samp(generated_data_sample[:, column_index], ori_data_sample[:, column_index])[0] for
         column_index in column_indexes])
