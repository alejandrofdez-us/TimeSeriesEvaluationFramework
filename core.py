from plots.dtw import generate_dtw_figures
from metrics.cc import cc
from metrics.cp import cp
from metrics.dtw import compute_dtw
from metrics.hi import hi
from metrics.js import js_distance_multivariate
from metrics.kl import kl_divergence
from metrics.ks import ks
from metrics.mmd import mmd_rbf
import numpy as np


def csv_has_header(filename):
    # TODO: ver si tiene header o no comprobando si en la primera fila hay algun valor númerico (por ejemplo cadenas mezcladas con números, en ese caso NO es fila de cabecera)
    np_array = np.loadtxt(filename, delimiter=",", max_rows=1)

    return False


def load_ts_from_csv(filename, has_header=None):
    if has_header is None:
        has_header = csv_has_header(filename)
    skiprows = 1 if has_header else 0
    return np.loadtxt(filename, delimiter=",", skiprows=skiprows)


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

def generate_figures(time_series_1, time_series_2, figures_to_be_generated):
    generated_figures = {}
    for figure_to_be_generated in figures_to_be_generated:
        generated_figures[figure_to_be_generated] = generate_figure(time_series_1, time_series_2,
                                                                    figure_to_be_generated)

    return generated_figures


def generate_figure(time_series_1, time_series_2, figure_to_be_generated):
    figure_function = get_figures_functions()
    generated_figures = figure_function[figure_to_be_generated](time_series_1, time_series_2)
    return generated_figures


def get_figures_functions():
    figures_functions = {
        'dtw': lambda ts_1, ts_2: generate_dtw_figures(ts_1, ts_2)
    }
    return figures_functions
