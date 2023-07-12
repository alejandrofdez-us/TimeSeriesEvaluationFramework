from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from plots.dimensionalty_reduction import DimensionalityReduction


class Tsne(DimensionalityReduction):
    @staticmethod
    def requires_all_samples():
        return True

    def __init__(self):
        super().__init__()

    def generate_figures(self, core, filename):
        super().generate_figures(core, filename)
        all_ts_prepared = np.concatenate((self.ts1_prepared, self.ts2_prepared), axis=0)
        colors = ["red" for _ in range(len(self.ts1_prepared))] + ["blue" for _ in range(len(self.ts2_prepared))]
        plot_array = []
        perplexities = {5, 10, min(40,
                                   len(self.ts2_prepared))}  # FIXME: Check what would be a valid perplexity when reduced number of samples. According to documentation "The perplexity must be less than the number of samples."
        for perplexity in perplexities:
            plot_array.append(self.__compute_tsne(len(self.ts1_prepared), colors, perplexity, all_ts_prepared, 300,
                                                  f'iter_300-perplexity_{perplexity}'))
            plot_array.append(self.__compute_tsne(len(self.ts1_prepared), colors, perplexity, all_ts_prepared, 1000,
                                                  f'iter_1000-perplexity_{perplexity}'))
        return plot_array

    def __compute_tsne(self, anal_sample_no, colors, perplexity, ts1_prepared_final, iterations, filename):
        tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=iterations)
        tsne_results = tsne.fit_transform(ts1_prepared_final)
        return self.__tsne_ploting(anal_sample_no, colors, tsne_results, filename)

    def __tsne_ploting(self, anal_sample_no, colors, tsne_results, filename):
        fig, ax = plt.subplots(1)
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="TS_1")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="TS_2")
        ax.legend()
        plt.title(f't_SNE_{filename}')
        plt.xlabel('x_tsne')
        plt.ylabel('y_tsne')
        plt.close("all")
        return fig, ax
