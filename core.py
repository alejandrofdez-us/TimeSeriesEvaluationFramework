from metrics.metric_factory import MetricFactory
from plots.plot_factory import PlotFactory

from plot_helper import update_figures_arguments
from window_sampler import create_ts1_ts2_associated_windows

TS1_TS2_ASSOCIATED_WINDOWS = None

# TODO: Crear objetos de configuracion de metrics y figures y pasarlos como argumentos (SE CREARIA EN MAIN)

# TODO: Reducir el número de argumentos
def compute_metrics(ts1, ts2_dict, stride, window_selection_metric, metrics_to_be_computed):
    global TS1_TS2_ASSOCIATED_WINDOWS
    if TS1_TS2_ASSOCIATED_WINDOWS is None:
        TS1_TS2_ASSOCIATED_WINDOWS = create_ts1_ts2_associated_windows(ts1, ts2_dict, stride, window_selection_metric)

    factory = MetricFactory(metrics_to_be_computed, TS1_TS2_ASSOCIATED_WINDOWS)
    #TODO: MetricFactory, devolver array de instancias de metricas
    #TODO: Iterar sobre la lista, computar cada métrica e ir componiendo el JSON
    computed_metrics = factory.get_metrics_json()

    return computed_metrics

# TODO: Reducir el número de argumentos
def generate_figures(ts1, ts2_dict, stride, window_selection_metric, header, figures_to_be_generated, ts_freq_secs):
    global TS1_TS2_ASSOCIATED_WINDOWS
    if TS1_TS2_ASSOCIATED_WINDOWS is None:
        TS1_TS2_ASSOCIATED_WINDOWS = create_ts1_ts2_associated_windows(ts1, ts2_dict, stride, window_selection_metric)

    args = update_figures_arguments(TS1_TS2_ASSOCIATED_WINDOWS, header, figures_to_be_generated, ts_freq_secs)

    factory = PlotFactory(figures_to_be_generated, args)
    generated_figures = factory.generate_figures()
    # TODO: Crear método que sea get_figures_that_requires_all_samples() que permita al main saber qué figuras requieren que se le pasen todas las muestras
    return generated_figures, factory.computed_figures_requires_all_samples
