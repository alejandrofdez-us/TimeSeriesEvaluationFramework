from metrics.metric_factory import MetricFactory

from helper import update_figures_arguments

from plots.dtw import generate_dtw_figures
from plots.tsne import generate_tsne_figures
from plots.pca import generate_pca_figures
from plots.deltas import generate_deltas_figures
from plots.evolution import generate_evolution_figures


def compute_metrics(time_series_1, time_series_2, metrics_to_be_computed):
    computed_metrics = {}

    factory = MetricFactory(metrics_to_be_computed, time_series_1, time_series_2)
    computed_metrics = factory.get_metrics_json()

    return computed_metrics

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
