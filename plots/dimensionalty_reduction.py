import numpy as np
from matplotlib import pyplot as plt

from plots.plot import Plot


class DimensionalityReduction(Plot):

    def __init__(self, fig_size=(8, 6)):
        super().__init__(fig_size)
        self.ts1_reduced_dimensions = None
        self.ts2_reduced_dimensions = None

    def initialize(self, core, filename):
        super().initialize(core, filename)
        self.ts1_reduced_dimensions, self.ts2_reduced_dimensions = self.reduce_tss_dimensionality(core.ts1_windows,
                                                                                                  np.asarray(core.ts2s))

    def reduce_tss_dimensionality(self, ts1, ts2):
        ts1_reduced = self.reduce_ts_dimensionality(ts1)
        ts2_reduced = self.reduce_ts_dimensionality(ts2)
        return ts1_reduced, ts2_reduced

    def reduce_ts_dimensionality(self, ts):
        seq_len = ts.shape[1]
        for i in range(len(ts)):
            if i == 0:
                ts_prepared = np.reshape(np.mean(ts[0, :, :], 1), [1, seq_len])
            else:
                ts_prepared = np.concatenate((ts_prepared, np.reshape(np.mean(ts[i, :, :], 1), [1, seq_len])))
        return ts_prepared

    def reduce_ts_dimensionality_refactored(self, ts):
        seq_len = ts.shape[1]
        set1_2d = ts.reshape(-1, 2)  # Conjunto de series temporales 1 en formato 2D
        for i in range(len(ts)):
            if i == 0:
                ts_prepared = np.reshape(np.mean(ts[0, :, :], 1), [1, seq_len])
            else:
                ts_prepared = np.concatenate((ts_prepared, np.reshape(np.mean(ts[i, :, :], 1), [1, seq_len])))
        return ts_prepared

    def compute(self, core, filename):
        assert len(
            core.ts2s) >= 2, f'The number of samples must be greater than 2 for {self.get_name()} analysis.'  # FIXME: comprobar si tsne necesita también al menos dos samples
        super().compute(core, filename)
