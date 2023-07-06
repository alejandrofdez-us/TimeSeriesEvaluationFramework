import csv
import importlib
import os
import numpy as np

def read_header_from_csv(filename, ts_delimiter, has_header):
    if has_header:
        header = np.genfromtxt(filename, delimiter=ts_delimiter, names=has_header, max_rows=1, dtype=str).dtype.names
    else:
        header = ["column-"+str(i) for i in range(len(header))]
    return header

def detect_line_delimiter(filename):
    with open(filename, "r", newline="") as file:
        ts_delimiter = csv.Sniffer().sniff(file.readline()).delimiter

    return ts_delimiter

def load_ts_from_csv(filename, has_header=None):
    ts_delimiter = detect_line_delimiter(filename)

    header = read_header_from_csv(filename, ts_delimiter, has_header)
    skiprows = 1 if has_header else 0

    return np.loadtxt(filename, delimiter=ts_delimiter, skiprows=skiprows), header


def load_ts_from_path(path, header_ts1, has_header=None):
    time_series = {}
    if os.path.isfile(path):
        ts2, header_ts2 = load_ts_from_csv(path, has_header)
        check_headers(header_ts1, header_ts2)
        time_series[os.path.basename(path)] = ts2
    else:
        for _, _, files in os.walk(path):
            for file in files:
                ts2, header_ts2 = load_ts_from_csv(f"{path}/{file}", has_header)
                check_headers(header_ts1, header_ts2)
                time_series[file] = ts2

    return time_series

def check_headers(header_ts1, header_ts2):
    if header_ts1 != header_ts2:
        raise ValueError("All time series must have the same header column names.")

def find_available_classes(folder_path, parent_class, package):
    available_classes = {}

    for _, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.py'):
                module_name = file_name[:-3]
                module = import_metric_module(module_name, package)
                if is_module_subclass(module, module_name, parent_class):
                    available_class = getattr(module, to_camel_case(module_name))()
                    available_classes[available_class.get_name()] = available_class
    return available_classes

def is_module_subclass(module, module_name, parent_class):
    if module is not None:
        class_name = to_camel_case(module_name)
        if parent_class.__name__ != module_name.capitalize():
            return hasattr(module, class_name) and issubclass(getattr(module, class_name), parent_class)
    return False

def import_metric_module(module_name, package):
    try:
        return importlib.import_module(f".{module_name}", package=package)
    except ImportError:
        return None

def to_camel_case(snake_str):
    components = snake_str.split('_')
    return "".join(x.capitalize() for x in components)
