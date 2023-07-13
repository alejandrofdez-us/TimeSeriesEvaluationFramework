import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from plots.dimensionalty_reduction import DimensionalityReduction


class Pca(DimensionalityReduction):
    @staticmethod
    def requires_all_samples():
        return True

    def compute(self, core, filename):
        assert len(core.ts1) > len(core.ts2s[0]), "TS1 sample size must be grater than the size of TS2 samples."
        super().compute(core, filename)
        pca = PCA(n_components=2)
        pca.fit(self.ts1_reduced_dimensions)
        pca_ts1 = pca.transform(self.ts1_reduced_dimensions)
        pca_ts2 = pca.transform(self.ts2_reduced_dimensions)
        fig, ax = self.generate_plot(pca_ts1, pca_ts2)
        return [(fig, ax)]

    def generate_plot(self, pca_ts1, pca_ts2):
        fig, ax = super().init_plot()
        colors = ["red" for _ in range(len(pca_ts1))] + ["blue" for _ in range(len(pca_ts2))]
        plt.scatter(pca_ts1[:, 0], pca_ts1[:, 1],
                    c=colors[:len(self.ts1_windows)], alpha=0.2, label="TS_1")
        plt.scatter(pca_ts2[:, 0], pca_ts2[:, 1],
                    c=colors[len(self.ts1_windows):], alpha=0.2, label="TS_2")
        super().set_labels('PCA', 'x_pca', 'y_pca')
        plt.close("all")
        return fig, ax
