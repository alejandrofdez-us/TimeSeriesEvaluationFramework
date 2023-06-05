from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def generate_tsne_figures(args):

    # Do t-SNE Analysis together
    prep_data_final = np.concatenate((args["prep_data"], args["prep_data_hat"]), axis=0)

    # TSNE analysis
    perplexity = 40
    # FIX perplexity when very few samples
    if args["n_samples"] < 40:
        perplexity = args["n_samples"]

    plot_array = []

    plot_array.append(compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 300,
                'iter_300-perplexity_40'))
    plot_array.append(compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 1000,
                'iter_1000-perplexity_40'))

    if args["n_samples"] >= 10:
        perplexity = 10
        plot_array.append(compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 300,
                    'iter_300-perplexity_10'))
        plot_array.append(compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 1000,
                    'iter_1000-perplexity_10'))

    if args["n_samples"] >= 5:
        perplexity = 5
        plot_array.append(compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 300,
                    'iter_300-perplexity_5'))
        plot_array.append(compute_tsne(args["anal_sample_no"], args["colors"], perplexity, prep_data_final, 1000,
                    'iter_1000-perplexity_5'))
    
    return plot_array
        
def compute_tsne(anal_sample_no, colors, perplexity, prep_data_final, iterations, filename):
    tsne_300 = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=iterations)
    tsne_results_300 = tsne_300.fit_transform(prep_data_final)

    return tsne_ploting(anal_sample_no, colors, tsne_results_300, filename)


def tsne_ploting(anal_sample_no, colors, tsne_results, filename):
    fig, ax = plt.subplots(1)
    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
    ax.legend()
    plt.title(f't-SNE plot {filename}')
    plt.xlabel('x_tsne')
    plt.ylabel('y_tsne')

    return fig, ax