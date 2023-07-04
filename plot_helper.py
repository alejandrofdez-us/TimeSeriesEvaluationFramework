import random
import numpy as np
import pandas as pd

def update_figures_arguments(time_series_2_dict, header, figures_to_be_generated, ts_freq_secs):
    are_tsne_pca_preprocessed = False
    args = {}
    tsne_pca_args = {}

    for filename, ts_dict in time_series_2_dict.items():
        args[filename] = {"ts1" : ts_dict["ts1"], "ts2" : ts_dict["ts2"], "header" : header}
        if ("tsne" in figures_to_be_generated or "pca" in figures_to_be_generated) and are_tsne_pca_preprocessed == False:
            tsne_pca_args = tsne_pca_preprocess(time_series_2_dict)
            args[filename].update(tsne_pca_args)
            are_tsne_pca_preprocessed = True
        else: 
            if are_tsne_pca_preprocessed:
                args[filename].update(tsne_pca_args)

        if "deltas" in figures_to_be_generated:
            args[filename].update(deltas_preprocess(ts_dict["ts1"], ts_freq_secs))

        if "evolution" in figures_to_be_generated:
            args[filename].update(evolution_preprocess(ts_dict["ts1"], ts_dict["ts2"], header))

    return args

def get_random_time_series_sample(seq_len, time_series):
    if len(time_series) > seq_len:
        ts_sample_start = random.randrange(0, len(time_series) - seq_len)
    else:
        ts_sample_start = 0

    ts_sample_end = ts_sample_start + seq_len
    ts_sample = time_series[ts_sample_start:ts_sample_end]
    return ts_sample

def deltas_preprocess(ts1, ts_freq_secs):
    args = {"seq_len" : len(ts1[:, 0]), "ts_freq_secs" : ts_freq_secs, "n_ts1_samples" : 1}

    return args

def evolution_preprocess(ts1, ts2, header):
    generated_data_sample_df = pd.DataFrame(ts2, columns=header)
    args = {"seq_len" : len(ts1[:, 0]), "ts_sample" : ts1, "generated_data_sample" : ts2,
                "generated_data_sample_df" : generated_data_sample_df}

    return args

def tsne_pca_preprocess (ts2_dict):
    generated_data = []
    ts1_for_visualization = []
    n_samples = 0
    seq_len = 0

    for ts_dict in ts2_dict.values():
        n_samples = n_samples + 1
        seq_len = len(ts_dict["ts2"])
        generated_data.append(ts_dict["ts2"])
        ts1_for_visualization.append(cut_and_mix_time_series(ts_dict["ts1"], seq_len)[0])

    args = {"ts1_data" : ts1_for_visualization, "gen_data" : generated_data, "n_samples" : n_samples}
    plot_args = tsne_pca_plot_preprocess(args)
    
    return plot_args

# TODO:Cortar y mezclar en dos metodos distintos, con un metodo shuffle_time_series que los llame
def cut_and_mix_time_series (ts, seq_len):
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ts) - seq_len + 1):
        _x = ts[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data

def tsne_pca_plot_preprocess (args):
    # Analysis sample size (for faster computation)
    anal_sample_no = min([args["n_samples"], len(args["ts1_data"])])
    idx = np.random.permutation(args["n_samples"])[:anal_sample_no]

    # Data preprocessing
    ts1 = np.asarray(args["ts1_data"])
    generated_data = np.asarray(args["gen_data"])

    ts1 = ts1[idx]
    generated_data = generated_data[idx]

    _, seq_len, _ = ts1.shape

    for i in range(anal_sample_no):
        if (i == 0):
            prep_data = np.reshape(np.mean(ts1[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(np.mean(generated_data[0, :, :], 1), [1, seq_len])
        else:
            prep_data = np.concatenate((prep_data,
                                        np.reshape(np.mean(ts1[i, :, :], 1), [1, seq_len])))
            prep_data_hat = np.concatenate((prep_data_hat,
                                            np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len])))

    colors = ["red" for _ in range(anal_sample_no)] + ["blue" for _ in range(anal_sample_no)]
    
    args.pop("ts1_data")
    args.pop("gen_data")

    args.update({"anal_sample_no" : anal_sample_no, "prep_data": prep_data, "prep_data_hat" : prep_data_hat, "colors" : colors})

    return args