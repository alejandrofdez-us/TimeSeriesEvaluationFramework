import json
from tqdm import tqdm

from core_config import CoreConfig
from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory
from plot_helper import update_figures_arguments
from window_sampler import create_ts1_ts2_associated_windows, split_ts_strided


class Core:
    def __init__(self, ts1, ts2s, ts2_names=None, header_names=None, core_config=None):
        self.ts1 = ts1
        self.ts2_dict = self.__build_ts2_dict(ts2s, ts2_names)
        self.core_config = core_config if core_config is not None else CoreConfig()
        self.ts1_windows = split_ts_strided(ts1, ts2s[0].shape[0], self.core_config.stride)
        self.ts1_ts2_associated_windows = create_ts1_ts2_associated_windows(self.ts1_windows, self.ts2_dict,
                                                                            self.core_config.window_selection_metric)
        self.metric_factory = MetricFactory(self.core_config.metric_config.metrics)
        self.plot_factory = PlotFactory.get_instance(self.core_config.plot_config.figures)
        self.header_names = header_names if header_names is not None else ["column-" + str(i) for i in
                                                                           range(ts1.shape[1])]

    def __build_ts2_dict(self, ts2s, ts2_names):
        ts2_names = ts2_names if ts2_names is not None else ["ts2-" + str(i) for i in range(len(ts2s))]
        return {ts2_name: ts2 for ts2, ts2_name in zip(ts2s, ts2_names)}

    def compute_metrics(self, show_progress=False):
        computed_metrics = {}
        iterator = self.__setup_progress_bar(self.ts1_ts2_associated_windows.items(), show_progress,
                                             'Computing metrics')
        for filename, ts_dict in iterator:
            computed_metrics[filename] = {}
            for metric_name, metric in self.metric_factory.metric_objects.items():
                if metric_name not in ts_dict["cached_metric"].keys():
                    computed_metrics[filename][metric_name] = metric.compute(ts_dict["ts1"], ts_dict["ts2"])
                else:
                    computed_metrics[filename][metric_name] = ts_dict["cached_metric"][metric_name]
            if show_progress:
                iterator.set_postfix(file=filename)
        computed_metrics = json.dumps(computed_metrics, indent=4)
        return computed_metrics

    def __setup_progress_bar(self, iterator, show_progress, desc):
        tqdm_iterator = iterator
        if show_progress:
            tqdm_iterator = tqdm(tqdm_iterator, total=len(iterator), desc=desc,
                                 disable=not show_progress)
        return tqdm_iterator

    def generate_figures(self, show_progress=False):
        args = update_figures_arguments(self.ts1_ts2_associated_windows, self.ts1_windows, self.header_names,
                                        self.core_config.plot_config)
        generated_figures = {}
        computed_figures_requires_all_samples = []
        iterator = self.__setup_progress_bar(args.items(), show_progress, 'Computing figures')
        for filename, figure_to_be_computed_args in iterator:
            generated_figures[filename] = {}
            for figure_name, figure in self.plot_factory.plots_to_be_generated.items():
                if figure_name not in computed_figures_requires_all_samples:
                    generated_figures[filename][figure_name] = figure.generate_figures(
                        figure_to_be_computed_args)
                    if figure_name in self.plot_factory.figures_requires_all_samples:
                        computed_figures_requires_all_samples.append(figure_name)
            if show_progress:
                iterator.set_postfix(file=filename)
        return generated_figures
