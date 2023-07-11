from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from plots.plot import Plot


class Tsne(Plot):
    @staticmethod
    def requires_all_samples():
        return True

    def __init__(self):
        super().__init__()
        self.ts1_prepared = None
        self.ts2_prepared = None

    def initialize(self, core, filename):
        super().initialize(core, filename)
        self.ts1_prepared = self.__prepare_ts_for_visualization(core.ts1_windows)
        self.ts2_prepared = self.__prepare_ts_for_visualization(np.asarray(core.ts2s))

    def __prepare_ts_for_visualization(self, ts):
        seq_len = ts.shape[1]
        for i in range(len(ts)):
            if i == 0:
                ts_prepared = np.reshape(np.mean(ts[0, :, :], 1), [1, seq_len])
            else:
                ts_prepared = np.concatenate((ts_prepared, np.reshape(np.mean(ts[i, :, :], 1), [1, seq_len])))
        return ts_prepared

    def generate_figures(self, core, filename):
        super().generate_figures(core, filename)
        all_ts_prepared = np.concatenate((self.ts1_prepared, self.ts2_prepared), axis=0)
        colors = ["red" for _ in range(len(self.ts1_prepared))] + ["blue" for _ in range(len(self.ts2_prepared))]
        plot_array = []
        perplexities = {5, 10, min(40, len(self.ts2_prepared))}
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
        plt.close()
        return fig, ax
