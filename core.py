from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory

from plot_helper import update_figures_arguments
from window_sampler import select_best_windows

def compute_metrics(ts1, ts2_dict, metrics_to_be_computed, stride, window_selection_metric):
    #cached_metric is a list containing the metric used to pick the best window in the first time series and the result of computing it    
    ts2_dict_windowed = select_best_windows(ts1, ts2_dict, stride, window_selection_metric)

    factory = MetricFactory(metrics_to_be_computed, ts2_dict_windowed)
    computed_metrics = factory.get_metrics_json()

    return computed_metrics

def generate_figures(ts1, ts2_dict, header, figures_to_be_generated, ts_freq_secs):
    args = update_figures_arguments(ts1, ts2_dict, header, figures_to_be_generated, ts_freq_secs)

    factory = PlotFactory(figures_to_be_generated, args)
    generated_figures = factory.generate_figures()

    return generated_figures
