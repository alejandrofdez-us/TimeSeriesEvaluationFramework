import numpy as np

from metrics.kl import Kl
from metrics.dtw import Dtw
from metrics.mmd import Mmd
from metrics.js import Js
from metrics.ks import Ks
from metrics.cc import Cc
from metrics.cp import Cp
from metrics.hi import Hi

def get_most_similar_ori_data_sample(ts1_windows, ts2, metric_object):
    computed_metric = float('inf')
    most_similar_sample = []
    for ts1 in ts1_windows:
        metric_object.compute_distance(ts1, ts2)
        current_distance = metric_object.compute_distance(ts1, ts2)
        if current_distance < computed_metric:
            computed_metric = current_distance
            most_similar_sample = ts1
    return most_similar_sample, computed_metric

def split_ori_data_strided(ori_data_df, seq_len, stride):
    assert seq_len <= ori_data_df.shape[0], 'seq_len cannot be greater than the original dataset length'
    if seq_len == ori_data_df.shape[0]:
        ori_data_windows_numpy = np.array([ori_data_df])
    else:
        start_sequence_range = list(range(0, ori_data_df.shape[0] - seq_len, stride))
        ori_data_windows_numpy = np.array(
            [ori_data_df[start_index:start_index + seq_len] for start_index in start_sequence_range])
    return ori_data_windows_numpy

def get_metric_function(windowing_metric):
    metric_functions = {
        "mmd": Mmd(),
        "dtw": Dtw(),
        "kl": Kl(),
        "js": Js(),
        "ks": Ks(),
        "cc": Cc(),
        "cp": Cp(),
        "hi": Hi(),    
    }

    return metric_functions[windowing_metric]

def select_best_windows(ts1, ts2_dict, stride, windowing_metric):
    metric_object = get_metric_function(windowing_metric)
    ts2_dict_windowed = {}

    for filename, ts2 in ts2_dict.items():
        ts1_windows = split_ori_data_strided(ts1, ts2.shape[0], stride)
        best_ts1, computed_metric = get_most_similar_ori_data_sample(ts1_windows, ts2, metric_object)
        cached_metric = [windowing_metric, computed_metric]

        if best_ts1.shape[1] < ts2.shape[1]:
            raise ValueError("The first time series must have equal or greater length than the second one.")

        ts2_dict_windowed[filename] = {}
        ts2_dict_windowed[filename]["ts1"] = best_ts1
        ts2_dict_windowed[filename]["ts2"] = ts2
        ts2_dict_windowed[filename]["cached_metric"] = cached_metric

    return ts2_dict_windowed
