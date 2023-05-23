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
import csv


def csv_has_header(filename, ts_delimiter, has_header):
    if has_header:
        header = np.genfromtxt(
            filename, delimiter=ts_delimiter, names=has_header, max_rows=1
        ).dtype.names
        for column in header:
            if any(char.isdigit() for char in str(column)):
                raise ValueError("Header must not contain numbers.")

    else:
        header = np.loadtxt(filename, delimiter=ts_delimiter, max_rows=1)
        header = ["" for _ in range(len(header))]

    return header


def detect_line_delimiter(filename):
    with open(filename, "r", newline="") as file:
        ts_delimiter = csv.Sniffer().sniff(file.read(1024)).delimiter

    return ts_delimiter


def load_ts_from_csv(filename, has_header=None):
    ts_delimiter = detect_line_delimiter(filename)

    header = csv_has_header(filename, ts_delimiter, has_header)
    skiprows = 1 if has_header else 0

    return np.loadtxt(filename, delimiter=ts_delimiter, skiprows=skiprows), header


def compute_metrics(time_series_1, time_series_2, metrics_to_be_computed):
    computed_metrics = {}

    for metric_to_be_computed in metrics_to_be_computed:
        computed_metrics[metric_to_be_computed] = compute_metric(
            time_series_1, time_series_2, metric_to_be_computed
        )

    return computed_metrics


def compute_metric(time_series_1, time_series_2, metric_to_be_computed):
    metric_function = get_metrics_functions()
    computed_metric = metric_function[metric_to_be_computed](
        time_series_1, time_series_2
    )

    return computed_metric


def get_metrics_functions():
    metric_functions = {
        "mmd": lambda ts_1, ts_2: mmd_rbf(ts_1, ts_2),
        "dtw": lambda ts_1, ts_2: compute_dtw(ts_1, ts_2),
        "kl": lambda ts_1, ts_2: kl_divergence(ts_1, ts_2),
        "js": lambda ts_1, ts_2: js_distance_multivariate(ts_1, ts_2),
        "ks": lambda ts_1, ts_2: ks(ts_1, ts_2),
        "cc": lambda ts_1, ts_2: cc(ts_1, ts_2),
        "cp": lambda ts_1, ts_2: cp(ts_1, ts_2),
        "hi": lambda ts_1, ts_2: hi(ts_1, ts_2),
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


def generate_figures(time_series_1, time_series_2, header, figures_to_be_generated):
    generated_figures = {}
    for figure_to_be_generated in figures_to_be_generated:
        generated_figures[figure_to_be_generated] = generate_figure(
            time_series_1, time_series_2, header, figure_to_be_generated
        )

    return generated_figures


def generate_figure(time_series_1, time_series_2, header, figure_to_be_generated):
    figure_function = get_figures_functions()
    generated_figures = figure_function[figure_to_be_generated](
        time_series_1, time_series_2, header
    )
    return generated_figures


def get_figures_functions():
    figures_functions = {
        "dtw": lambda ts_1, ts_2, header: generate_dtw_figures(ts_1, ts_2, header)
    }
    return figures_functions
