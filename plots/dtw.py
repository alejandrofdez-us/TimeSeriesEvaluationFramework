from dtaidistance import dtw_visualisation
from dtaidistance import dtw


def generate_dtw_figure(time_series_1_column, time_series_2_column):
    path = dtw.warping_path(time_series_1_column, time_series_2_column)
    figure = dtw_visualisation.plot_warping(time_series_1_column, time_series_2_column, path)

    return figure


def generate_dtw_figures(time_series_1, time_series_2):
    figures = []
    for column in range(time_series_1.shape[1]):
        figures.append(generate_dtw_figure(time_series_1[:, column].reshape(-1, 1),
                                           time_series_2[:, column].reshape(-1, 1)))
    return figures
