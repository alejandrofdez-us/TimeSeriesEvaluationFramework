from dtaidistance import dtw_visualisation
from dtaidistance import dtw


def generate_dtw_figure(time_series_1_column, time_series_2_column, column):
    path = dtw.warping_path(time_series_1_column, time_series_2_column)
    figure = dtw_visualisation.plot_warping(
        time_series_1_column, time_series_2_column, path
    )

    figure[0].axes[0].set_title(column)

    return figure


def generate_dtw_figures(time_series_1, time_series_2, header):
    figures = []
    for column, index in zip(header, range(len(header))):
        figures.append(
            generate_dtw_figure(
                time_series_1[:, index], time_series_2[:, index], column
            )
        )
    return figures
