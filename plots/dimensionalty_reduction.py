import numpy as np
from matplotlib import pyplot as plt

from plots.plot import Plot


class DimensionalityReduction(Plot):

    def __init__(self, plot_size=(8, 6)):
        super().__init__()
        self.ts1_prepared = None
        self.ts2_prepared = None
        plt.rcParams["figure.figsize"] = plot_size

    def initialize(self, core, filename):
        super().initialize(core, filename)
        self.ts1_prepared, self.ts2_prepared = self.reduce_tss_dimensionality(core.ts1_windows,
                                                                              np.asarray(core.ts2s))

    def reduce_ts_dimensionality(self, ts):
        seq_len = ts.shape[1]
        for i in range(len(ts)):
            if i == 0:
                ts_prepared = np.reshape(np.mean(ts[0, :, :], 1), [1, seq_len])
            else:
                ts_prepared = np.concatenate((ts_prepared, np.reshape(np.mean(ts[i, :, :], 1), [1, seq_len])))
        return ts_prepared

    def reduce_tss_dimensionality(self, ts1, ts2):
        ts1_reduced = self.reduce_ts_dimensionality(ts1)
        ts2_reduced = self.reduce_ts_dimensionality(ts2)
        return ts1_reduced, ts2_reduced

    def reduce_ts_dimensionality_refactored(self, ts):
        seq_len = ts.shape[1]
        set1_2d = ts.reshape(-1, 2)  # Conjunto de series temporales 1 en formato 2D
        for i in range(len(ts)):
            if i == 0:
                ts_prepared = np.reshape(np.mean(ts[0, :, :], 1), [1, seq_len])
            else:
                ts_prepared = np.concatenate((ts_prepared, np.reshape(np.mean(ts[i, :, :], 1), [1, seq_len])))
        return ts_prepared
