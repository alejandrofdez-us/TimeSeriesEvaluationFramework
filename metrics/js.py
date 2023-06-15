from metrics.kl import kl_divergence, kl_divergence_univariate

def js (X,Y):
    metric_result = f"Multivariate: {js_distance_multivariate(X,Y)}"

    for column in range(Y.shape[1]):
        metric_result = metric_result + f" Column {column}: {js_distance(Y[:, column].reshape(-1, 1), X[:, column].reshape(-1, 1))}"

    return metric_result

def js_distance(array_1, array_2, num_bins=100):
    KL_p_m, KL_q_m = kl_divergence_univariate(array_1, array_2, num_bins=num_bins)
    JS_p_q = (KL_p_m + KL_q_m) / 2
    return JS_p_q


def js_distance_multivariate(array_1, array_2):
    kl_diverenge_1 = kl_divergence(array_1, array_2)
    kl_diverenge_2 = kl_divergence(array_2, array_1)
    return (kl_diverenge_1 + kl_diverenge_2) / 2
