from tqdm import tqdm

from metrics.MetricComputer import MetricComputer
from core_config import CoreConfig
from metrics.metric_factory import MetricFactory
from plots.PlotComputer import PlotComputer
from plots.plot_factory import PlotFactory
from window_sampler import create_ts1_ts2_associated_windows, split_ts_strided


class Core:
    def __init__(self, ts1, ts2s, core_config=None):
        self.ts1 = ts1
        self.ts2s = ts2s
        self.core_config = core_config if core_config is not None else CoreConfig()
        self.ts2_dict = self.__build_ts2_dict(ts2s, core_config.ts2_names)
        self.ts1_windows = split_ts_strided(ts1, ts2s[0].shape[0], self.core_config.stride)
        self.ts1_ts2_associated_windows = create_ts1_ts2_associated_windows(self.ts1_windows, self.ts2_dict,
                                                                            self.core_config.window_selection_metric)
        self.metric_factory = MetricFactory.get_instance(self.core_config.metric_config.metrics)
        self.plot_factory = PlotFactory.get_instance(self.core_config.plot_config.figures)  # FIXME: eliminar?
        self.header_names = core_config.header_names if core_config.header_names is not None else ["column-" + str(i)
                                                                                                   for i in
                                                                                                   range(ts1.shape[1])]

    def __build_ts2_dict(self, ts2s, ts2_filenames):
        ts2_filenames = ts2_filenames if ts2_filenames is not None else ["ts2-" + str(i) for i in range(len(ts2s))]
        return {ts2_name: ts2 for ts2, ts2_name in zip(ts2s, ts2_filenames)}

    def get_metric_computer(self):
        return MetricComputer(self.ts1_ts2_associated_windows, self.metric_factory.metric_objects)

    def get_plot_computer(self):
        return PlotComputer(self, self.ts1_ts2_associated_windows, self.plot_factory.plots_to_be_generated)
