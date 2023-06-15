import scipy
import statistics

def ks (X,Y):
    metric_result = f"Multivariate: {compute_ks(X,Y)}"

    for column in range(Y.shape[1]):
        metric_result = metric_result + f" Column {column}: {compute_ks(Y[:, column].reshape(-1, 1), X[:, column].reshape(-1, 1))}"

    return metric_result

def compute_ks(generated_data_sample, ori_data_sample):
    column_indexes = range(generated_data_sample.shape[1])
    return statistics.mean(
        [scipy.stats.ks_2samp(generated_data_sample[:, column_index], ori_data_sample[:, column_index])[0] for
         column_index in column_indexes])
