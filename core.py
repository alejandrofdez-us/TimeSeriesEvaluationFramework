from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory

from plot_helper import update_figures_arguments
from window_sampler import create_ts1_ts2_associated_windows

TS1_TS2_ASSOCIATED_WINDOWS = None

def compute_metrics(ts1, ts2_dict, metric_config):
    global TS1_TS2_ASSOCIATED_WINDOWS
    if TS1_TS2_ASSOCIATED_WINDOWS is None:
        TS1_TS2_ASSOCIATED_WINDOWS = create_ts1_ts2_associated_windows(ts1, ts2_dict, metric_config.stride, metric_config.window_selection_metric)

    factory = MetricFactory(metric_config.metrics, TS1_TS2_ASSOCIATED_WINDOWS)
    #TODO: MetricFactory, devolver array de instancias de metricas
    #TODO: Iterar sobre la lista, computar cada métrica e ir componiendo el JSON
    computed_metrics = factory.get_metrics_json()

    return computed_metrics

def generate_figures(ts1, ts2_dict, header, plot_config):
    global TS1_TS2_ASSOCIATED_WINDOWS
    if TS1_TS2_ASSOCIATED_WINDOWS is None:
        TS1_TS2_ASSOCIATED_WINDOWS = create_ts1_ts2_associated_windows(ts1, ts2_dict, plot_config.stride, plot_config.window_selection_metric)

    args = update_figures_arguments(TS1_TS2_ASSOCIATED_WINDOWS, header, plot_config.figures, plot_config.timestamp_frequency_seconds)

    factory = PlotFactory(plot_config.figures, args)
    generated_figures = factory.generate_figures()

    return generated_figures
