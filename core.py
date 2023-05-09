from metrics.cc import cc
from metrics.cp import cp
from metrics.dtw import compute_dtw
from metrics.hi import hi
from metrics.js import js_distance_multivariate
from metrics.kl import kl_divergence
from metrics.ks import ks
from metrics.mmd import mmd_rbf


def compute_metrics(time_series_1, time_series_2, metrics_to_be_computed):
    computed_metrics = {}

    for metric_to_be_computed in metrics_to_be_computed:
        computed_metrics[metric_to_be_computed] = compute_metric(time_series_1, time_series_2, metric_to_be_computed)

    return computed_metrics


def compute_metric(time_series_1, time_series_2, metric_to_be_computed):
    metric_function = get_metrics_functions()
    computed_metric = metric_function[metric_to_be_computed](time_series_1, time_series_2)

    return computed_metric


def get_metrics_functions():
    metric_functions = {
        'mmd': lambda ts_1, ts_2: mmd_rbf(ts_1, ts_2),
        'dtw': lambda ts_1, ts_2: compute_dtw(ts_1, ts_2),
        'kl': lambda ts_1, ts_2: kl_divergence(ts_1, ts_2),
        'js': lambda ts_1, ts_2: js_distance_multivariate(ts_1, ts_2),
        'ks': lambda ts_1, ts_2: ks(ts_1, ts_2),
        'cc': lambda ts_1, ts_2: cc(ts_1, ts_2),
        'cp': lambda ts_1, ts_2: cp(ts_1, ts_2),
        'hi': lambda ts_1, ts_2: hi(ts_1, ts_2)
    }
    return metric_functions

# def get_metrics_functions_by_columns ():
#     metric_functions = {
#         'mmd': lambda ts_1, ts_2: mmd_rbf(ts_1, ts_2),
#         'dtw': lambda ts_1, ts_2: compute_dtw(ts_1, ts_2),
#         'kl': lambda ts_1, ts_2: kl_divergence_univariate(ts_1, ts_2)[0],
#         'js': lambda ts_1, ts_2: JSdistance(ts_1, ts_2),
#         'ks': lambda ts_1, ts_2: compute_ks(ts_1, ts_2),
#         'cc': lambda ts_1, ts_2: compute_cc(ts_1, ts_2),
#         'cp': lambda ts_1, ts_2: compute_cp(ts_1, ts_2),
#         'hi': lambda ts_1, ts_2: compute_hi(ts_1, ts_2)
#     }
#     return metric_functions
