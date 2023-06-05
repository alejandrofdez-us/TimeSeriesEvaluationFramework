def deltas_preprocess(ts1):

    args = {"seq_len" : len(ts1[:, 0]), "timestamp_frequency_secs" : 300, "n_ori_samples" : 1}

    return args
