from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory

from plot_helper import update_figures_arguments
from window_sampler import select_best_window

def compute_metrics(ts1, ts2, metrics_to_be_computed, stride, window_selection_metric):
    #cached_metric is a list containing the metric used to pick the best window in the first time series and the result of computing it    
    ts1_best_window, cached_metric = select_best_window(ts1, ts2, stride, window_selection_metric)

    if ts1.shape[1] < ts2.shape[1]:
        raise ValueError("The first time series must have equal or greater length than the second one.")

    factory = MetricFactory(metrics_to_be_computed, ts1_best_window, ts2)
    computed_metrics = factory.get_metrics_json(cached_metric)

    return computed_metrics

def generate_figures(ts1, ts2, header, figures_to_be_generated, ts_freq_secs):
    args = update_figures_arguments(ts1, ts2, header, figures_to_be_generated, ts_freq_secs)

    factory = PlotFactory(figures_to_be_generated, args)
    generated_figures = factory.generate_figures()

    return generated_figures
