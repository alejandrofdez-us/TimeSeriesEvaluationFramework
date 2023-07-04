import os
import numpy as np

from metrics.metric_factory import MetricFactory

def get_most_similar_ts_sample(ts1_windows, ts2, metric_object):
    current_best = float('inf')
    most_similar_sample = []
    for ts1_window in ts1_windows:
        current_distance = metric_object.compute_distance(ts1_window, ts2)
        if metric_object.compare(current_distance, current_best) > 0:
            current_best = current_distance
            most_similar_sample = ts1_window

    return most_similar_sample, current_best

def split_ts_strided(ts_np, seq_len, stride):
    assert seq_len <= ts_np.shape[0], 'seq_len cannot be greater than the original dataset length'
    assert (ts_np.shape[0] - seq_len) >= stride-1, 'stride cannot be greater than the size difference between time series'
    start_sequence_range = list(range(0, (ts_np.shape[0] - seq_len)+1, stride))
    ts_windows = np.array(
        [ts_np[start_index:start_index + seq_len] for start_index in start_sequence_range])
    return ts_windows

def get_metric_function(window_selection_metric):
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics")
    metric_classes = MetricFactory.find_metrics_in_directory([window_selection_metric], folder_path)

    return metric_classes[window_selection_metric]

def create_ts1_ts2_associated_windows(ts1, ts2_dict, stride, window_selection_metric):
    metric_object = get_metric_function(window_selection_metric)
    ts1_ts2_associated_windows = {}

    for filename, ts2 in ts2_dict.items():
        ts1_windows = split_ts_strided(ts1, ts2.shape[0], stride)
        best_ts1, computed_metric = get_most_similar_ts_sample(ts1_windows, ts2, metric_object)
        cached_metric = [window_selection_metric, computed_metric]

        ts1_ts2_associated_windows[filename] = {}
        ts1_ts2_associated_windows[filename]["ts1"] = best_ts1
        ts1_ts2_associated_windows[filename]["ts2"] = ts2
        ts1_ts2_associated_windows[filename]["cached_metric"] = cached_metric

    return ts1_ts2_associated_windows
