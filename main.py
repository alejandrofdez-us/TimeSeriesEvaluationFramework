import os
import argparse
from core import compute_metrics, generate_figures
from reader import load_ts_from_csv, load_ts_from_path


def main(arguments):
    try:
        ts1, header_ts1 = load_ts_from_csv(
            arguments.time_series_1_filename, arguments.header
        )
        ts2_dict = load_ts_from_path(
            arguments.time_series_2_path, header_ts1, arguments.header
        )

        if arguments.metrics:
            computed_metrics = compute_metrics(ts1, ts2_dict, arguments.stride, arguments.window_selection_metric, arguments.metrics)
            save_metrics(computed_metrics)

        if arguments.figures:
            generated_figures, computed_figures_requires_all_samples = generate_figures(ts1, ts2_dict, arguments.stride, arguments.window_selection_metric, header_ts1, arguments.figures, arguments.timestamp_frequency_seconds)
            # TODO: Se hace llamada al get_figures_that_requires_all_samples() para saber qué figuras requieren que se le pasen todas las muestras
            save_figures(generated_figures, computed_figures_requires_all_samples)

    except ValueError as error:
        print("Error: ", error)

def save_figures(figures_dict, computed_figures_requires_all_samples, path="results/figures"):
    for filename, figures in figures_dict.items():
        for figure_name, plots in figures.items():
            for plot in plots:
                plot_label = plot[0].axes[0].get_title()
                dir_path = ""

                if figure_name not in computed_figures_requires_all_samples:
                    original_filename = filename.split(".")[0]
                    dir_path = f"{path}/{original_filename}/{figure_name}/"
                else:
                    dir_path = f"{path}/{figure_name}/"
                os.makedirs(dir_path, exist_ok=True)

                plot[0].savefig(
                    f"{dir_path}{plot_label}.pdf",
                    format="pdf", bbox_inches="tight"
                )

def save_metrics(computed_metrics, path="results/metrics"):
    os.makedirs(f"{path}", exist_ok=True)
    with open(f"{path}/results.json", "w") as file:
        file.write(computed_metrics)


if __name__ == "__main__":
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
        "-ts2_path",
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
        #TODO: Crear lista de metricas disponibles llamando a MetricFactory.get_available_metrics()
        choices=["js", "mmd", "dtw", "kl", "ks", "cc", "cp", "hi"],
        required=False,
    )
    parser.add_argument(
        "-f",
        "--figures",
        nargs="+",
        help="<Optional> Include figure names to be generated as a list separated by spaces.",
        #TODO: Crear lista de figuras disponibles llamando a PlotFactory.get_available_figures()
        choices=["tsne", "pca", "dtw", "evolution", "deltas"],
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
