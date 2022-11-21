import random
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
    for ori_data_sample in tqdm(ori_data_windows_numpy):
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
                # "cpu_util_percent": {
                #     "column_index": 0,
                #     "y_axis_min": 0,
                #     "y_axis_max": 100
                # },
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
                    # "cpu_util_percent": {
                    #     "type": "numerical",
                    #     "subtype": "float"
                    # },
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
