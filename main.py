import argparse

import numpy as np

from core import compute_metrics


def main(time_series1, time_series2, metrics):
    computed_metrics = compute_metrics(time_series1, time_series2, metrics)
    print(computed_metrics)

    # TODO: Incorporar las figuras de manera parecida a como hemos hecho con las métricas númericas
    # TODO: Pensar si computar las métricas comparando columna a columna y no su version multivariate
    # TODO: Pensar como  guardar los resultados (consola o fichero y formato csv? json? html? varios?), pedirle al
    #  usuario nombre de fichero resultante y que por defecto sea algo así como results.csv


if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser(
        usage='python main.py -ts1 path_to_file_1 -ts2 path_to_file_2 --metrics js mmd... [--figures] tsne pca...')
    parser.add_argument('-ts1', '--time_series_1_filename',
                        help='<Required> Include a csv filename that represents a time-series including a header.',
                        type=str,
                        required=True)
    parser.add_argument('-ts2', '--time_series_2_filename',
                        help='<Required> Include a csv filename that represents a time-series including a header.',
                        type=str,
                        required=True)
    parser.add_argument('-m', '--metrics', nargs='+',
                        help='<Required> Include metrics to be computed as a list separated by spaces.',
                        choices=['js', 'mmd', 'dtw', 'kl', 'ks', 'cc', 'cp', 'hi'],
                        required=True)
    parser.add_argument('-f', '--figures', nargs='+',
                        help='<Required> Include figure names to be generated as a list separated by spaces.',
                        choices=['tsne', 'pca', 'dtw', 'evolution', 'deltas'],
                        required=False)

    args = parser.parse_args()
    ts1 = np.loadtxt(args.time_series_1_filename, delimiter=",", skiprows=1)
    ts2 = np.loadtxt(args.time_series_2_filename, delimiter=",", skiprows=1)

    main(ts1, ts2, args.metrics)
