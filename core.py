from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory

from helper import update_figures_arguments

def compute_metrics(time_series_1, time_series_2, metrics_to_be_computed, computed_chosen_metric):
    factory = MetricFactory(metrics_to_be_computed, time_series_1, time_series_2)
    computed_metrics = factory.get_metrics_json(computed_chosen_metric)

    return computed_metrics

def generate_figures(time_series_1, time_series_2, header, figures_to_be_generated, ts_freq_secs):
    args = update_figures_arguments(time_series_1, time_series_2, header, figures_to_be_generated, ts_freq_secs)

    factory = PlotFactory(figures_to_be_generated, args)
    generated_figures = factory.generate_figures()

    return generated_figures
