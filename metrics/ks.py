import scipy
import statistics

from metrics.metric import Metric

class KS(Metric):
    def compute(self, ts1, ts2):
        metric_result = {"Multivariate": self.__compute_ks(ts1, ts2)}

        for column in range(ts2.shape[1]):
            metric_result.update({f"Column {column}": self.__compute_ks(ts1[:, column].reshape(-1, 1), ts2[:, column].reshape(-1, 1))})

        return metric_result

    def __compute_ks(self, ts1, ts2):
        column_indexes = range(ts2.shape[1])
        return statistics.mean(
            [scipy.stats.ks_2samp(ts2[:, column_index], ts1[:, column_index])[0] for
            column_index in column_indexes])

# TODO: Las métricas deberían devolver un objeto JSON que pudiera ser procesado de manera automática y qué se hace con estos resultados será responsabilidad de la herramienta de consola u otras herramientas que puedan usar el core.
# TODO: Ejemplo de JSON:
# {'ks':{
#     'multivariate': XX.YY,
#     'univariate': [
#         {'column_0': ZZ.WW},
#         {'column_1': UU.II},
#         {'column_N': QQ.WW}
#     ]
# }}

