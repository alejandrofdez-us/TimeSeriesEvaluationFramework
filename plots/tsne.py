from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

from plots.plot import Plot

class Tsne(Plot):
    def generate_figures(self, args):
        plot_array = self.__generate_tsne_figures(args)
        return plot_array

    @staticmethod
    def requires_all_samples():
        return True

    def __generate_tsne_figures(self, args):
        prep_data_final = np.concatenate((args["prep_data"], args["prep_data_hat"]), axis=0)

        perplexity = 40

        if args["n_samples"] < 40:
            perplexity = args["n_samples"]

        plot_array = []

        plot_array.append(self.__compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 300,
                    'iter_300-perplexity_40'))
        plot_array.append(self.__compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 1000,
                    'iter_1000-perplexity_40'))

        if args["n_samples"] >= 10:
            perplexity = 10
            plot_array.append(self.__compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 300,
                        'iter_300-perplexity_10'))
            plot_array.append(self.__compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 1000,
                        'iter_1000-perplexity_10'))

        if args["n_samples"] >= 5:
            perplexity = 5
            plot_array.append(self.__compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 300,
                        'iter_300-perplexity_5'))
            plot_array.append(self.__compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 1000,
                        'iter_1000-perplexity_5'))

        return plot_array

    def __compute_tsne(self, anal_sample_no, colors, perplexity, prep_data_final, iterations, filename):
        tsne_300 = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=iterations)
        tsne_results_300 = tsne_300.fit_transform(prep_data_final)

        return self.__tsne_ploting(anal_sample_no, colors, tsne_results_300, filename)

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
