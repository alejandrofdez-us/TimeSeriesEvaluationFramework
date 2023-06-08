import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_pca_figures(args):
    # PCA Analysis
    assert len(args["prep_data"]) >= 2, 'The number of samples must be greater than 2 for PCA analysis.'    
    pca = PCA(n_components=2)
    pca.fit(args["prep_data"])
    pca_results = pca.transform(args["prep_data"])
    pca_hat_results = pca.transform(args["prep_data_hat"])

    # Plotting
    fig, ax = plt.subplots(1)
    plt.scatter(pca_results[:, 0], pca_results[:, 1],
                c=args["colors"][:args["anal_sample_no"]], alpha=0.2, label="Time Series 1")
    plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1],
                c=args["colors"][:args["anal_sample_no"]], alpha=0.2, label="Time Series 2")

    ax.legend()
    plt.title('PCA')
    plt.xlabel('x_pca')
    plt.ylabel('y_pca')

    plot_array = []
    plot_array.append((fig, ax))

    return plot_array