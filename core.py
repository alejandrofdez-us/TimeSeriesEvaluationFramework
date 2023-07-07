import json
from metrics.metric_config import MetricConfig
from metrics.metric_factory import MetricFactory
from plots.plot_config import PlotConfig
from plots.plot_factory import PlotFactory
from plot_helper import update_figures_arguments
from window_sampler import create_ts1_ts2_associated_windows, split_ts_strided


class Core:
    def __init__(self, ts1, ts2s, ts2_names=None, header_names=None, metric_config=None, plot_config=None):
        self.ts1 = ts1
        self.ts2_dict = self.__build_ts2_dict(ts2s, ts2_names)
        self.metric_config = metric_config if metric_config is not None else MetricConfig()
        self.plot_config = plot_config if plot_config is not None else PlotConfig()
        self.ts1_windows = split_ts_strided(ts1, ts2s[0].shape[0], self.metric_config.stride)
        self.ts1_ts2_associated_windows = create_ts1_ts2_associated_windows(self.ts1_windows, self.ts2_dict,
                                                                            self.metric_config.window_selection_metric)
        self.metric_factory = MetricFactory(self.metric_config.metrics)
        self.plot_factory = PlotFactory(self.plot_config.figures)
        self.header_names = header_names if header_names is not None else ["column-" + str(i) for i in
                                                                           range(ts1.shape[1])]

    def __build_ts2_dict(self, ts2s, ts2_names):
        ts2_names = ts2_names if ts2_names is not None else ["ts2-" + str(i) for i in range(len(ts2s))]
        return {ts2_name: ts2 for ts2, ts2_name in zip(ts2s, ts2_names)}

    def compute_metrics(self):
        computed_metrics = {}
        for filename, ts_dict in self.ts1_ts2_associated_windows.items():
            computed_metrics[filename] = {}
            for metric_name, metric in self.metric_factory.metric_objects.items():
                if metric_name not in ts_dict["cached_metric"].keys():
                    computed_metrics[filename][metric_name] = metric.compute(ts_dict["ts1"], ts_dict["ts2"])
                else:
                    computed_metrics[filename][metric_name] = ts_dict["cached_metric"][metric_name]
        computed_metrics = json.dumps(computed_metrics, indent=4)
        return computed_metrics

    def generate_figures(self):
        args = update_figures_arguments(self.ts1_ts2_associated_windows, self.ts1_windows, self.header_names,
                                        self.plot_config)
        generated_figures = {}
        computed_figures_requires_all_samples = []
        for filename, figure_to_be_computed_args in args.items():
            generated_figures[filename] = {}
            for figure_name, figure in self.plot_factory.plots_to_be_generated.items():
                if figure_name not in computed_figures_requires_all_samples:
                    generated_figures[filename][figure_name] = figure.generate_figures(
                        figure_to_be_computed_args)
                    if figure_name in self.plot_factory.figures_requires_all_samples:
                        computed_figures_requires_all_samples.append(figure_name)
        return generated_figures
