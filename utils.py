import random
import re

from dtaidistance import dtw_ndim
import numpy as np
from tqdm import tqdm


def get_ori_data_sample(seq_len, ori_data):
    if len(ori_data) > seq_len:
        ori_data_sample_start = random.randrange(0, len(ori_data) - seq_len)
    else: # seq_len is the full ori_data_length
        ori_data_sample_start = 0

    ori_data_sample_end = ori_data_sample_start + seq_len
    ori_data_sample = ori_data[ori_data_sample_start:ori_data_sample_end]
    return ori_data_sample

def get_most_similar_ori_data_sample(ori_data_windows_numpy, generated_data_sample):
    minimum_dtw = float('inf')
    most_similar_sample = []
    #most_similar_sample = [(dtw_ndim.distance_fast(ori_data_sample, generated_data_sample),ori_data_sample) for ori_data_sample in ori_data_windows_numpy].min()[1]
    progress_bar_search = tqdm(ori_data_windows_numpy, desc='Searching minimum dtw distance from original data', colour='yellow', position=2, leave=False)
    for ori_data_sample in progress_bar_search:
        current_distance = dtw_ndim.distance_fast(ori_data_sample, generated_data_sample)
        if current_distance < minimum_dtw:
            minimum_dtw = current_distance
            most_similar_sample = ori_data_sample

    return most_similar_sample

def split_ori_data_strided(ori_data_df, seq_len, stride):
    start_sequence_range = list(range(0, ori_data_df.shape[0] - seq_len, stride))
    ori_data_windows_numpy = np.array([ori_data_df[start_index:start_index+seq_len] for start_index in start_sequence_range])
    return ori_data_windows_numpy

def get_dataset_info(trace_name):
    if trace_name == 'alibaba2018':
        dataset_info = {
            "timestamp_frequency_secs": 300,
            "column_config": {
                "cpu_util_percent": {
                    "column_index": 0,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "mem_util_percent": {
                    "column_index": 1,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "net_in": {
                    "column_index": 2,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "net_out": {
                    "column_index": 3,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "disk_io_percent": {
                    "column_index": 4,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                }

            },
            "metadata": {
                "fields": {
                    "cpu_util_percent": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "mem_util_percent": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "net_in": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "net_out": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "disk_io_percent": {
                        "type": "numerical",
                        "subtype": "float"
                    }
                }
            }
        }
    if trace_name == 'alibaba2018-4columns':
        dataset_info = {
            "timestamp_frequency_secs": 10,
            "column_config": {
                "mem_util_percent": {
                    "column_index": 0,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "net_in": {
                    "column_index": 1,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "net_out": {
                    "column_index": 2,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                },
                "disk_io_percent": {
                    "column_index": 3,
                    "y_axis_min": 0,
                    "y_axis_max": 100
                }

            },
            "metadata": {
                "fields": {
                    "mem_util_percent": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "net_in": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "net_out": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "disk_io_percent": {
                        "type": "numerical",
                        "subtype": "float"
                    }
                }
            }
        }
    elif trace_name == 'google2019':
        dataset_info = {
            "timestamp_frequency_secs": 300,
            "column_config": {
                "cpu": {
                    "column_index": 0,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "mem": {
                    "column_index": 1,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "assigned_mem": {
                    "column_index": 2,
                    "y_axis_min": 0,
                    "y_axis_max": 1
                },
                "cycles_per_instruction": {
                    "column_index": 3
                }
            },
            "metadata": {
                "fields": {
                    "cpu": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "mem": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "assigned_mem": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "cycles_per_instruction": {
                        "type": "numerical",
                        "subtype": "float"
                    }
                }
            }
        }
    elif trace_name == 'azure_v2':
        dataset_info = {
            "timestamp_frequency_secs": 300,
            "column_config": {
                "cpu_total": {
                    "column_index": 0
                },
                "mem_total": {
                    "column_index": 1
                }
            },
            "metadata": {
                "fields": {
                    "cpu_total": {
                        "type": "numerical",
                        "subtype": "float"
                    },
                    "mem_total": {
                        "type": "numerical",
                        "subtype": "float"
                    }
                }
            }
        }
    elif trace_name == 'reddit':
        dataset_info = {
            "timestamp_frequency_secs": 3600,
            "column_config": {
                "interactions": {
                    "column_index": 0
                }
            }
        }
    return dataset_info


def normalize_start_time_to_zero(sample):
    timestamp_column = sample[:, 0]
    min_timestamp = np.min(timestamp_column)
    normalized_timestamp_column = timestamp_column - min_timestamp
    sample[:, 0] = normalized_timestamp_column
    return sample


def extract_experiment_parameters(saved_experiment_parameters):
    saved_experiment_parameters_dict = dict(
        item.split("=") for item in re.split(', (?![^\[]*\])', saved_experiment_parameters.replace('Namespace(', '').replace('Parameters(', '').replace(')', '').replace('\n', '')))
    parameters_values = ''
    parameters_keys = ''
    for parameter_value in saved_experiment_parameters_dict.values():
        parameters_values += parameter_value + ';'
    for parameter_key in saved_experiment_parameters_dict.keys():
        parameters_keys += parameter_key + ';'
    return parameters_keys, parameters_values, saved_experiment_parameters_dict


def save_metrics(avg_results, metrics_results, path_to_save_metrics, saved_experiments_parameters, saved_metrics):
    _, _, parameters_dict = extract_experiment_parameters(saved_experiments_parameters)
    if 'data_name' in parameters_dict:
        data_name = parameters_dict['data_name']
    elif 'trace' in parameters_dict:
        data_name = parameters_dict['trace']
    else:
        data_name = 'not_found'
    if 'iteration' in parameters_dict:
        iteration = parameters_dict['iteration']
    elif 'epochs' in parameters_dict:
        iteration = parameters_dict['epochs']
    else:
        iteration = 'not_found'

    seq_len = parameters_dict['seq_len']
    with open(
            path_to_save_metrics + '/metrics-' + data_name + '-iterations-' + iteration + '-seq_len' + seq_len + '.txt',
            'w') as f:
        f.write(saved_experiments_parameters + '\n\n')
        f.write(saved_metrics + '\n\n')
        f.write(repr(avg_results) + '\n')
        computed_metrics, metrics_values = results_for_excel(avg_results)
        f.write(
            'Results of the following metrics: ' + computed_metrics + ' in spanish locale Excel format:' + '\n' + metrics_values + '\n')
        f.write(repr(metrics_results))
    #print("Metrics saved in file", f.name)
    return computed_metrics, metrics_values


def results_for_excel(avg_results):
    metrics_values = ''
    computed_metrics = ''
    for metric_name in avg_results:
        computed_metrics += metric_name + ';'
        metrics_values += str(avg_results[metric_name]).replace('.', ',') + ';'

    return computed_metrics, metrics_values
