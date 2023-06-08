from dtaidistance import dtw_visualisation
from dtaidistance import dtw


def generate_dtw_figure(time_series_1_column, time_series_2_column, column):
    path = dtw.warping_path(time_series_1_column, time_series_2_column)
    figure = dtw_visualisation.plot_warping(
        time_series_1_column, time_series_2_column, path
    )

    figure[0].axes[0].set_title(f"DTW_{column}")

    return figure


def generate_dtw_figures(args):
    figures = []
    for column, index in zip(args["header"], range(len(args["header"]))):
        figures.append(
            generate_dtw_figure(
                args["ts1"][:, index], args["ts2"][:, index], column
            )
        )
    return figures
