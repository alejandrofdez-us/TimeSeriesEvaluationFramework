from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory

from plot_helper import update_figures_arguments
from window_sampler import select_best_windows

def generate_metrics_and_figures(ts1, ts2_dict, metrics_to_be_computed, stride, window_selection_metric, header, figures_to_be_generated, ts_freq_secs):
    generated_figures = None

    #cached_metric is a list containing the metric used to pick the best window in the first time series and the result of computing it    
    ts2_dict_windowed = select_best_windows(ts1, ts2_dict, stride, window_selection_metric)

    computed_metrics = compute_metrics(ts2_dict_windowed, metrics_to_be_computed)
    
    if figures_to_be_generated != None:
        generated_figures = generate_figures(ts2_dict_windowed, header, figures_to_be_generated, ts_freq_secs)
    
    return computed_metrics, generated_figures

def compute_metrics(ts2_dict_windowed, metrics_to_be_computed):
    factory = MetricFactory(metrics_to_be_computed, ts2_dict_windowed)
    computed_metrics = factory.get_metrics_json()

    return computed_metrics

def generate_figures(ts2_dict_windowed, header, figures_to_be_generated, ts_freq_secs):
    args = update_figures_arguments(ts2_dict_windowed, header, figures_to_be_generated, ts_freq_secs)

    factory = PlotFactory(figures_to_be_generated, args)
    generated_figures = factory.generate_figures()

    return generated_figures
