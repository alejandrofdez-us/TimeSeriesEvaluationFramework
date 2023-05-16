import argparse
import numpy as np
from core import compute_metrics, generate_figures


def main(arguments):
    ts1 = np.loadtxt(arguments.time_series_1_filename, delimiter=",", skiprows=1)
    ts2 = np.loadtxt(arguments.time_series_2_filename, delimiter=",", skiprows=1)
    # TODO: Comprobar que ambas series temporales tienen el mismo número de variables (columnas)

    computed_metrics = compute_metrics(ts1, ts2, arguments.metrics)
    print(computed_metrics)

    if arguments.figures:
        figures = generate_figures(ts1, ts2, arguments.figures)
        save_figures(figures)

    # TODO: Incorporar las figuras de manera parecida a como hemos hecho con las métricas númericas
    # TODO: Pensar si computar las métricas comparando columna a columna y no su version multivariate
    # TODO: Pensar como  guardar los resultados (consola o fichero y formato csv? json? html? varios?), pedirle al
    #  usuario nombre de fichero resultante y que por defecto sea algo así como results.csv
    # TODO: ver si hay alguna manera de empaquetarlo para que no necesite instalar los requirements.txt


def save_figures(figures_dict, path='/figures'):
    for figure_name, figures in figures_dict.items():
        # TODO: Comprobar si en cada clave del diccionario tenemos un array de figuras (por ejemplo dtw) o bien una única figura (PCA)
        i = 0
        for figure in figures:
            figure[0].savefig(f'{i}-{figure_name}.pdf', format='pdf')
            i = i + 1


if __name__ == '__main__':
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
    main(args)
