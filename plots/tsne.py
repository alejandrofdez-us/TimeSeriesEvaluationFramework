import itertools

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from plots.dimensionalty_reduction import DimensionalityReduction


class Tsne(DimensionalityReduction):
    @staticmethod
    def requires_all_samples():
        return True

    def compute(self, core, filename):
        super().compute(core, filename)
        ts1_ts2_reduced_dimensions_concatenated = np.concatenate(
            (self.ts1_reduced_dimensions, self.ts2_reduced_dimensions), axis=0)
        colors = ["red" for _ in range(len(self.ts1_reduced_dimensions))] + ["blue" for _ in
                                                                             range(len(self.ts2_reduced_dimensions))]
        plot_array = []
        perplexities = [5, 10, min(40, len(self.ts1_reduced_dimensions))]
        perplexities = list(filter(lambda p: p <= len(self.ts1_reduced_dimensions), perplexities))
        iterations = [300, 1000]

        for perplexity, n_iterations in itertools.product(perplexities, iterations):
            tsne_embedding = self.__compute_tsne(perplexity,
                                                 ts1_ts2_reduced_dimensions_concatenated, n_iterations)
            tsne_plot = self.__tsne_ploting(len(self.ts1_reduced_dimensions), colors, tsne_embedding,
                                            f'iter_{n_iterations}-perplexity_{perplexity}')
            plot_array.append(tsne_plot)
        return plot_array

    def __compute_tsne(self, perplexity, ts1_ts2_reduced_dimensions_concatenated, iterations):
        tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=iterations)
        return tsne.fit_transform(ts1_ts2_reduced_dimensions_concatenated)

    def __tsne_ploting(self, anal_sample_no, colors, tsne_embedding, title):
        plt.rcParams["figure.figsize"] = self.fig_size
        fig, ax = plt.subplots(1)
        plt.scatter(tsne_embedding[:anal_sample_no, 0], tsne_embedding[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="TS_1")
        plt.scatter(tsne_embedding[anal_sample_no:, 0], tsne_embedding[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="TS_2")
        ax.legend()
        plt.title(f't_SNE_{title}')
        plt.xlabel('x_tsne')
        plt.ylabel('y_tsne')
        plt.close("all")
        return fig, ax
