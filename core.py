import json

from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory

from plot_helper import update_figures_arguments
from window_sampler import create_ts1_ts2_associated_windows

TS1_TS2_ASSOCIATED_WINDOWS = None

def compute_metrics(ts1, ts2_dict, metric_config):
    global TS1_TS2_ASSOCIATED_WINDOWS
    if TS1_TS2_ASSOCIATED_WINDOWS is None:
        TS1_TS2_ASSOCIATED_WINDOWS = create_ts1_ts2_associated_windows(ts1, ts2_dict, metric_config.stride, metric_config.window_selection_metric)

    factory = MetricFactory(metric_config.metrics)

    computed_metrics = {}
    for filename, ts_dict in TS1_TS2_ASSOCIATED_WINDOWS.items():
        computed_metrics[filename] = {}
        for metric_to_be_computed in factory.metrics_to_be_computed:
            metric = factory.create_metric(metric_to_be_computed)
            computed_metrics[filename][metric_to_be_computed] = metric.compute(
                ts_dict["ts1"], ts_dict["ts2"], ts_dict["cached_metric"]
            )

    computed_metrics = json.dumps(computed_metrics, indent=4)

    return computed_metrics

def generate_figures(ts1, ts2_dict, header, plot_config):
    global TS1_TS2_ASSOCIATED_WINDOWS
    if TS1_TS2_ASSOCIATED_WINDOWS is None:
        TS1_TS2_ASSOCIATED_WINDOWS = create_ts1_ts2_associated_windows(ts1, ts2_dict, plot_config.stride, plot_config.window_selection_metric)

    args = update_figures_arguments(TS1_TS2_ASSOCIATED_WINDOWS, header, plot_config.figures, plot_config.timestamp_frequency_seconds)

    factory = PlotFactory(plot_config.figures, args)

    generated_figures = {}
    for filename, figure_to_be_computed_args in args.items():
        generated_figures[filename] = {}
        for figure_to_be_computed in factory.figures_to_be_generated:
            if (figure_to_be_computed not in factory.computed_figures_requires_all_samples):
                plot = factory.create_figure(figure_to_be_computed)
                generated_figures[filename][figure_to_be_computed] = plot.generate_figures(
                    figure_to_be_computed_args
                )
                if figure_to_be_computed in factory.figures_requires_all_samples:
                    factory.computed_figures_requires_all_samples.append(figure_to_be_computed)


    return generated_figures
