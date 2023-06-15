from metrics.cc import cc
from metrics.cp import cp
from metrics.dtw import dtw
from metrics.hi import hi
from metrics.js import js
from metrics.kl import kl
from metrics.ks import ks
from metrics.mmd import mmd_rbf

from helper import update_figures_arguments

from plots.dtw import generate_dtw_figures
from plots.tsne import generate_tsne_figures
from plots.pca import generate_pca_figures
from plots.deltas import generate_deltas_figures
from plots.evolution import generate_evolution_figures


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
        "dtw": lambda ts_1, ts_2: dtw(ts_1, ts_2),
        "kl": lambda ts_1, ts_2: kl(ts_1, ts_2),
        "js": lambda ts_1, ts_2: js(ts_1, ts_2),
        "ks": lambda ts_1, ts_2: ks(ts_1, ts_2),
        "cc": lambda ts_1, ts_2: cc(ts_1, ts_2),
        "cp": lambda ts_1, ts_2: cp(ts_1, ts_2),
        "hi": lambda ts_1, ts_2: hi(ts_1, ts_2),
    }
    return metric_functions

def generate_figures(time_series_1, time_series_2, header, figures_to_be_generated, ts_freq_secs):
    generated_figures = {}
    args = update_figures_arguments(time_series_1, time_series_2, header, figures_to_be_generated, ts_freq_secs)

    for figure_to_be_generated in figures_to_be_generated:
        generated_figures[figure_to_be_generated] = generate_figure(
            args, figure_to_be_generated
        )

    return generated_figures


def generate_figure(args, figure_to_be_generated):
    figure_function = get_figures_functions()
    generated_figures = figure_function[figure_to_be_generated](args)
    return generated_figures


def get_figures_functions():
    figures_functions = {
        "dtw": lambda args: generate_dtw_figures(args),
        "tsne": lambda args: generate_tsne_figures(args),
        "pca": lambda args: generate_pca_figures(args),
        "deltas": lambda args: generate_deltas_figures(args),
        "evolution": lambda args: generate_evolution_figures(args)
    }
    return figures_functions
