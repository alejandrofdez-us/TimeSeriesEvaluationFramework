import numpy as np
from plots.plot import Plot


class DimensionalityReduction(Plot):
    def __init__(self, fig_size=(8, 6)):
        super().__init__(fig_size)
        self.ts1_reduced_dimensions = None
        self.ts2_reduced_dimensions = None

    def _initialize(self, similarity_ts, ts2_filename):
        super()._initialize(similarity_ts, ts2_filename)
        self.ts1_reduced_dimensions, self.ts2_reduced_dimensions = self.__reduce_tss_dimensionality(
            similarity_ts.ts1_windows,
            np.asarray(similarity_ts.ts2s))

    def __reduce_tss_dimensionality(self, ts1, ts2):
        ts1_reduced = self.__reduce_ts_dimensionality(ts1)
        ts2_reduced = self.__reduce_ts_dimensionality(ts2)
        return ts1_reduced, ts2_reduced

    def __reduce_ts_dimensionality(self, ts):
        seq_len = ts.shape[1]
        for i in range(len(ts)):
            if i == 0:
                ts_prepared = np.reshape(np.mean(ts[0, :, :], 1), [1, seq_len])
            else:
                ts_prepared = np.concatenate((ts_prepared, np.reshape(np.mean(ts[i, :, :], 1), [1, seq_len])))
        return ts_prepared

    # FIXME: Que hacemos cone esto?
    def __reduce_ts_dimensionality_refactored(self, ts):  # FIXME: valorar el uso del que tenemos u otro nuevo
        seq_len = ts.shape[1]
        set1_2d = ts.reshape(-1, 2)  # Conjunto de series temporales 1 en formato 2D
        for i in range(len(ts)):
            if i == 0:
                ts_prepared = np.reshape(np.mean(ts[0, :, :], 1), [1, seq_len])
            else:
                ts_prepared = np.concatenate((ts_prepared, np.reshape(np.mean(ts[i, :, :], 1), [1, seq_len])))
        return ts_prepared

    def _generate_colors(self, color1_size, color2_size):
        return ['red' for _ in range(color1_size)] + ['blue' for _ in range(color2_size)]
