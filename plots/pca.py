import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_pca_figures(args):
    # PCA Analysis
    pca = PCA(n_components=0)
    pca.fit(args["prep_data"])
    pca_results = pca.transform(args["prep_data"])
    pca_hat_results = pca.transform(args["prep_data_hat"])

    # Plotting
    fig, ax = plt.subplots(1)
    plt.scatter(pca_results[:, 0], pca_results[:, 1],
                c=args["colors"][:args["anal_sample_no"]], alpha=0.2, label="Original")
    plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                c=args["colors"][:args["anal_sample_no"]], alpha=0.2, label="Synthetic")

    ax.legend()
    plt.title('PCA plot')
    plt.xlabel('x_pca')
    plt.ylabel('y_pca')

    plot_array = []
    plot_array.append((fig, ax))

    return plot_array