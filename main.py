import os
import argparse
from tqdm import tqdm

from core_config import CoreConfig
from metrics.metric_config import MetricConfig
from metrics.metric_factory import MetricFactory
from plots.plot_config import PlotConfig
from plots.plot_factory import PlotFactory
from core import Core
from csv_reader_helper import load_ts_from_csv, load_ts_from_path


def main(arguments):
    try:
        ts1, header_ts1 = load_ts_from_csv(arguments.time_series_1_filename, arguments.header)
        ts2_dict = load_ts_from_path(arguments.time_series_2_path, header_ts1, arguments.header)
        core_config = __create_core_config(arguments, list(ts2_dict.keys()), header_ts1)
        core = Core(ts1, list(ts2_dict.values()), core_config)
        if core_config.metric_config.metrics:
            computed_metrics = core.compute_metrics(show_progress=True)
            __save_metrics(computed_metrics)
        if core_config.plot_config.figures:
            generated_figures = core.generate_figures(show_progress=True)
            __save_figures(generated_figures, show_progress=True)
    except ValueError as error:
        print("Error: ", error)


def __create_core_config(arguments, ts2_names, header_names):
    metric_config = None
    plot_config = None if arguments.timestamp_frequency_seconds is None else PlotConfig(None,
                                                                                        arguments.timestamp_frequency_seconds)
    if arguments.metrics is not None or arguments.figures is not None:
        metric_config = MetricConfig(arguments.metrics) if arguments.metrics else MetricConfig([])
        plot_config = PlotConfig(arguments.figures,
                                 arguments.timestamp_frequency_seconds) if arguments.figures else PlotConfig([],
                                                                                                             arguments.timestamp_frequency_seconds)
    core_config = CoreConfig(metric_config, plot_config, arguments.stride, arguments.window_selection_metric, ts2_names,
                             header_names)
    return core_config


def __save_figures(figures_dict, path="results/figures", show_progress=False):
    iterator = figures_dict.items()
    if show_progress:
        iterator = tqdm(figures_dict.items(), total=len(figures_dict.items()), desc='   Saving figures',
                        disable=not show_progress)
    for filename, figures in iterator:
        for figure_name, plots in figures.items():
            for plot in plots:
                plot_label = plot[0].axes[0].get_title()
                if figure_name in PlotFactory.get_instance().figures_requires_all_samples:
                    dir_path = f"{path}/{figure_name}/"
                else:
                    original_filename = filename.split(".")[0]
                    dir_path = f"{path}/{original_filename}/{figure_name}/"
                os.makedirs(dir_path, exist_ok=True)
                plot[0].savefig(f"{dir_path}{plot_label}.pdf", format="pdf", bbox_inches="tight")
        if show_progress:
            iterator.set_postfix(file=filename)


def __save_metrics(computed_metrics, path="results/metrics"):
    os.makedirs(f"{path}", exist_ok=True)
    with open(f"{path}/results.json", "w") as file:
        file.write(computed_metrics)


if __name__ == "__main__":
    available_metrics = MetricFactory.find_available_metrics().keys()
    available_figures = PlotFactory.find_available_figures().keys()
    parser = argparse.ArgumentParser(
        usage="python main.py -ts1 path_to_file_1 -ts2_path path_to_files_2 [--metrics] [js ...] [--figures] [tsne ...] \
            [--header] [--timestamp_frequency_seconds] 300 [--stride] 2 [--window_selection_metric] metric_name"
    )
    parser.add_argument(
        "-ts1",
        "--time_series_1_filename",
        help="<Required> Include a csv filename that represents a time series. If ts1 is bigger than time series in ts2_path, it will be splitted in windows.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ts2",
        "--time_series_2_path",
        help="<Required> Include the path to a csv file or a directory with csv files, each one representing time series.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        help="<Optional> Include metrics to be computed as a list separated by spaces.",
        choices=available_metrics,
        required=False,
    )
    parser.add_argument(
        "-f",
        "--figures",
        nargs="+",
        help="<Optional> Include figure names to be generated as a list separated by spaces.",
        choices=available_figures,
        required=False,
    )
    parser.add_argument(
        "-head",
        "--header",
        help="<Optional> If the time-series includes a header row.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-ts_freq_secs",
        "--timestamp_frequency_seconds",
        help="<Optional> Include the frequency in seconds in which samples were taken.",
        required=False,
        default=1,
        type=int,
    )
    parser.add_argument(
        "-strd",
        "--stride",
        help="<Optional> Include the stride to be used in moving windows over samples.",
        required=False,
        default=1,
        type=int,
    )
    parser.add_argument(
        "-w_select_met",
        "--window_selection_metric",
        help="<Optional> Include the chosen metric used to pick the best window in the first time series.",
        required=False,
        default="dtw",
        type=str,
    )
    args = parser.parse_args()
    main(args)
