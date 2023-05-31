import os
import argparse
from core import compute_metrics, generate_figures, load_ts_from_csv


def main(arguments):
    try:
        ts1, header_ts1, ts_delimiter = load_ts_from_csv(
            arguments.time_series_1_filename, arguments.header
        )
        ts2, header_ts2, _ = load_ts_from_csv(
            arguments.time_series_2_filename, arguments.header
        )

        if ts1.shape[1] != ts2.shape[1]:
            raise ValueError("Both time series must have the same number of columns.")

        elif header_ts1 != header_ts2:
            raise ValueError("Both time series must have the same header column names.")

        computed_metrics = compute_metrics(ts1, ts2, arguments.metrics)
        print(computed_metrics)

        if arguments.figures:
            figures = generate_figures(ts1, ts2, header_ts1, arguments.figures, ts_delimiter)
        save_figures(figures)

    except ValueError as error:
        print("Error: ", error)

    # TODO: Incorporar las figuras de manera parecida a como hemos hecho con las métricas númericas
    # TODO: Pensar si computar las métricas comparando columna a columna y no su version multivariate
    # TODO: Pensar como  guardar los resultados (consola o fichero y formato csv? json? html? varios?), pedirle al
    #  usuario nombre de fichero resultante y que por defecto sea algo así como results.csv
    # ? TODO: ver si hay alguna manera de empaquetarlo para que no necesite instalar los requirements.txt -> Que necesitamos empaquetar?


def save_figures(figures_dict, path="figures"):
    for figure_name, figures in figures_dict.items():
        for figure in figures:
            figure_label = figure[0].axes[0].get_title()
            os.makedirs(f"{path}/{figure_name}", exist_ok=True)

            figure[0].savefig(
                f"{path}/{figure_name}/{figure_label}.pdf",
                format="pdf",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="python main.py -ts1 path_to_file_1 -ts2 path_to_file_2 --metrics js mmd... [--figures] tsne pca... [--header]"
    )
    parser.add_argument(
        "-ts1",
        "--time_series_1_filename",
        help="<Required> Include a csv filename that represents a time-series including a header.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-ts2",
        "--time_series_2_filename",
        help="<Required> Include a csv filename that represents a time-series including a header.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--metrics",
        nargs="+",
        help="<Required> Include metrics to be computed as a list separated by spaces.",
        choices=["js", "mmd", "dtw", "kl", "ks", "cc", "cp", "hi"],
        required=True,
    )
    parser.add_argument(
        "-f",
        "--figures",
        nargs="+",
        help="<Optional> Include figure names to be generated as a list separated by spaces.",
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

    args = parser.parse_args()
    main(args)
