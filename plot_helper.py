import numpy as np
import pandas as pd
import random

def update_figures_arguments(time_series_1, time_series_2_dict, header, figures_to_be_generated, ts_freq_secs):
    args = {"ts1" : time_series_1, "ts2_dict" : time_series_2_dict, "header" : header}

    if "tsne" in figures_to_be_generated or "pca" in figures_to_be_generated:
        args.update(tsne_pca_preprocess(time_series_1, time_series_2_dict)) # ver

    if "deltas" in figures_to_be_generated:
        args.update(deltas_preprocess(time_series_1, ts_freq_secs))
    
    if "evolution" in figures_to_be_generated:
        args.update(evolution_preprocess(time_series_1, time_series_2_dict, header)) # ver

    return args

def get_ori_data_sample(seq_len, ori_data):
    if len(ori_data) > seq_len:
        ori_data_sample_start = random.randrange(0, len(ori_data) - seq_len)
    else:  # seq_len is the full ori_data_length
        ori_data_sample_start = 0

    ori_data_sample_end = ori_data_sample_start + seq_len
    ori_data_sample = ori_data[ori_data_sample_start:ori_data_sample_end]
    return ori_data_sample

def deltas_preprocess(ts1, ts_freq_secs):

    args = {"seq_len" : len(ts1[:, 0]), "ts_freq_secs" : ts_freq_secs, "n_ori_samples" : 1}

    return args

def evolution_preprocess(ts1, ts2_dict, header):

    generated_data_samples_df = {}
    for filename, ts2 in ts2_dict.items():
        generated_data_samples_df[filename] = pd.DataFrame(ts2, columns=header)
    args = {"seq_len" : len(ts1[:, 0]), "ori_data_sample" : ts1, "generated_data_samples" : ts2_dict,
                "generated_data_samples_df" : generated_data_samples_df}

    return args

def tsne_pca_preprocess (ts1, ts2_dict):
    generated_data = []
    ori_data_for_visualization = []
    n_samples = 0
    seq_len = 0

    for ts2 in ts2_dict.values():
        n_samples = n_samples + 1
        seq_len = len(ts2)
        generated_data.append(ts2)
        ori_data_for_visualization.append(seq_cut_and_mix(ts1, seq_len))

    args = {"ori_data" : ori_data_for_visualization, "gen_data" : generated_data, "n_samples" : n_samples}
    plot_args = plot_preprocess(args)
    
    return plot_args
        
def seq_cut_and_mix (ori_data, seq_len):
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len + 1):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data

def plot_preprocess (args):
    # Analysis sample size (for faster computation)
    anal_sample_no = min([args["n_samples"], len(args["ori_data"])])
    idx = np.random.permutation(args["n_samples"])[:anal_sample_no]

    # Data preprocessing
    ori_data = np.asarray(args["ori_data"])
    generated_data = np.asarray(args["gen_data"])

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    _, seq_len, _ = ori_data.shape

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
    colors = ["red" for _ in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]
    
    args.pop("ori_data")
    args.pop("gen_data")

    args.update({"anal_sample_no" : anal_sample_no, "prep_data": prep_data, "prep_data_hat" : prep_data_hat, "colors" : colors})

    return args