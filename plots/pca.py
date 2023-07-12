import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from plots.dimensionalty_reduction import DimensionalityReduction


class Pca(DimensionalityReduction):
    @staticmethod
    def requires_all_samples():
        return True

    def __init__(self):
        super().__init__()

    def compute(self, core, filename):
        super().compute(core, filename)
        pca = PCA(n_components=2)
        pca.fit(self.ts1_reduced_dimensions)
        pca_ts1 = pca.transform(self.ts1_reduced_dimensions)
        pca_ts2 = pca.transform(self.ts2_reduced_dimensions)
        fig, ax = self.generate_plot(pca_ts1, pca_ts2)
        return [(fig, ax)]

    def generate_plot(self, pca_ts1, pca_ts2):
        fig, ax = plt.subplots(1)
        colors = ["red" for _ in range(len(pca_ts1))] + ["blue" for _ in range(len(pca_ts2))]
        plt.scatter(pca_ts1[:, 0], pca_ts1[:, 1],
                    c=colors[:len(self.ts1_windows)], alpha=0.2, label="TS_1")
        plt.scatter(pca_ts2[:, 0], pca_ts2[:, 1],
                    c=colors[len(self.ts1_windows):], alpha=0.2, label="TS_2")
        ax.legend()
        plt.title('PCA')
        plt.xlabel('x_pca')
        plt.ylabel('y_pca')
        plt.close("all")
        return fig, ax
