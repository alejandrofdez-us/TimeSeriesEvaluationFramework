import random
import numpy as np
import pandas as pd


def update_figures_arguments(time_series_2_dict, ts1_windows, header, plot_config):
    are_tsne_pca_preprocessed = False
    args = {}
    tsne_pca_args = {}
    for filename, ts_dict in time_series_2_dict.items():
        args[filename] = {"ts1": ts_dict["ts1"], "ts2": ts_dict["ts2"], "header": header}
        # FIXME: refactor to make all these checks strings dynamic and adapt to the new object oriented config classes.
        if ("tsne" in plot_config.figures or "pca" in plot_config.figures) and are_tsne_pca_preprocessed is False:
            tsne_pca_args = tsne_pca_preprocess(time_series_2_dict, ts1_windows)
            args[filename].update(tsne_pca_args)
            are_tsne_pca_preprocessed = True
        else:
            if are_tsne_pca_preprocessed:
                args[filename].update(tsne_pca_args)

        if "delta" in plot_config.figures:
            args[filename].update(delta_preprocess(ts_dict["ts1"], plot_config.timestamp_frequency_seconds))

        if "2d" in plot_config.figures:
            args[filename].update(two_dimensions_preprocess(ts_dict["ts1"], ts_dict["ts2"], header))
    return args


def get_random_time_series_sample(seq_len, time_series):
    if len(time_series) > seq_len:
        ts_sample_start = random.randrange(0, len(time_series) - seq_len)
    else:
        ts_sample_start = 0
    ts_sample_end = ts_sample_start + seq_len
    ts_sample = time_series[ts_sample_start:ts_sample_end]
    return ts_sample


def delta_preprocess(ts1, ts_freq_secs):
    args = {"seq_len": len(ts1[:, 0]), "ts_freq_secs": ts_freq_secs, "n_ts1_samples": 1}
    return args


def two_dimensions_preprocess(ts1, ts2, header):
    generated_data_sample_df = pd.DataFrame(ts2, columns=header)
    args = {"seq_len": len(ts1[:, 0]), "ts_sample": ts1, "generated_data_sample": ts2,
            "generated_data_sample_df": generated_data_sample_df}
    return args


def tsne_pca_preprocess(ts2_dict, ts1_windows):
    generated_data = [ts_dict['ts2'] for ts_dict in list(ts2_dict.values())]
    ts1_for_visualization = [np.array(subarray) for subarray in ts1_windows]
    args = {"ts1_data": ts1_for_visualization, "gen_data": generated_data, "n_samples": len(ts2_dict.keys())}
    plot_args = tsne_pca_plot_preprocess(args)
    return plot_args


def cut_time_series(ts, seq_len):
    temp_data = []
    for i in range(0, len(ts) - seq_len + 1):
        _x = ts[i:i + seq_len]
        temp_data.append(_x)

    return temp_data


def mix_time_series(temp_data):
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


def shuffle_time_series(ts, seq_len):
    temp_data = cut_time_series(ts, seq_len)
    data = mix_time_series(temp_data)

    return data


def tsne_pca_plot_preprocess(args):
    anal_sample_no = min([args["n_samples"], len(args["ts1_data"])])
    idx = np.random.permutation(args["n_samples"])[:anal_sample_no]

    ts1 = np.asarray(args["ts1_data"])
    generated_data = np.asarray(args["gen_data"])

    ts1 = ts1[idx]
    generated_data = generated_data[idx]

    _, seq_len, _ = ts1.shape

    for i in range(anal_sample_no):
        if i == 0:
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

    args.update(
        {"anal_sample_no": anal_sample_no, "prep_data": prep_data, "prep_data_hat": prep_data_hat, "colors": colors})

    return args
