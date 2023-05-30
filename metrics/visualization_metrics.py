"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def visualization(ori_data, generated_data, analysis, path_for_saving_images, n_samples=150):
    """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
    - n_samples: n_samples to analyze
  """
    # Analysis sample size (for faster computation)
    anal_sample_no = min([n_samples, len(ori_data)])
    idx = np.random.permutation(n_samples)[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]

    if analysis == 'pca':
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        f, ax = plt.subplots(1)
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")

        ax.legend()
        plt.title('PCA plot')
        plt.xlabel('x_pca')
        plt.ylabel('y_pca')
        plt.savefig(path_for_saving_images + '/PCA.png')



    elif analysis == 'tsne':

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE analysis
        perplexity = 40
        # FIX perplexity when very few samples
        if n_samples < 40:
            perplexity = n_samples

        compute_tsne(anal_sample_no, colors, path_for_saving_images, perplexity, prep_data_final, 300,
                     'iter_300-perplexity_40')
        compute_tsne(anal_sample_no, colors, path_for_saving_images, perplexity, prep_data_final, 1000,
                     'iter_1000-perplexity_40')

        perplexity = 10
        compute_tsne(anal_sample_no, colors, path_for_saving_images, perplexity, prep_data_final, 300,
                     'iter_300-perplexity_10')
        compute_tsne(anal_sample_no, colors, path_for_saving_images, perplexity, prep_data_final, 1000,
                     'iter_1000-perplexity_10')

        perplexity = 5
        compute_tsne(anal_sample_no, colors, path_for_saving_images, perplexity, prep_data_final, 300,
                     'iter_300-perplexity_5')
        compute_tsne(anal_sample_no, colors, path_for_saving_images, perplexity, prep_data_final, 1000,
                     'iter_1000-perplexity_5')


def compute_tsne(anal_sample_no, colors, path_for_saving_images, perplexity, prep_data_final, iterations, filename):
    tsne_300 = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=iterations)
    tsne_results_300 = tsne_300.fit_transform(prep_data_final)
    tsne_ploting(anal_sample_no, colors, path_for_saving_images, tsne_results_300, filename)


def tsne_ploting(anal_sample_no, colors, path_for_saving_images, tsne_results, filename):
    f, ax = plt.subplots(1)
    plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1],
                c=colors[:anal_sample_no], alpha=0.2, label="Original")
    plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1],
                c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
    ax.legend()
    plt.title(f't-SNE plot {filename}')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig(f'{path_for_saving_images}/t-SNE-{filename}.png')
