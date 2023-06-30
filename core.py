from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory

from plot_helper import update_figures_arguments
from window_sampler import create_ts1_ts2_associated_windows

ts1_ts2_associated_windows = None

def compute_metrics(ts1, ts2_dict, stride, window_selection_metric, metrics_to_be_computed):
    global ts1_ts2_associated_windows
    if ts1_ts2_associated_windows is None:
        ts1_ts2_associated_windows = create_ts1_ts2_associated_windows(ts1, ts2_dict, stride, window_selection_metric)

    factory = MetricFactory(metrics_to_be_computed, ts1_ts2_associated_windows)
    computed_metrics = factory.get_metrics_json()

    return computed_metrics

def generate_figures(ts1, ts2_dict, stride, window_selection_metric, header, figures_to_be_generated, ts_freq_secs):
    global ts1_ts2_associated_windows
    if ts1_ts2_associated_windows is None:
        ts1_ts2_associated_windows = create_ts1_ts2_associated_windows(ts1, ts2_dict, stride, window_selection_metric)

    args = update_figures_arguments(ts1_ts2_associated_windows, header, figures_to_be_generated, ts_freq_secs)

    factory = PlotFactory(figures_to_be_generated, args)
    generated_figures = factory.generate_figures()

    return generated_figures, factory.computed_figures_requires_all_samples
