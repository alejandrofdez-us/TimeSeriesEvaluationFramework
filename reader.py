import csv
import re
import os
import numpy as np

def csv_has_header(filename, ts_delimiter, has_header):
    if has_header:
        header = np.genfromtxt(filename, delimiter=ts_delimiter, names=has_header, max_rows=1, dtype=str).dtype.names

        if header_has_numeric(header):
            raise ValueError("Header must not contain numeric values.")

    else:
        header = np.loadtxt(filename, delimiter=ts_delimiter, max_rows=1)
        header = ["column-"+str(i) for i in range(len(header))]

    return header

def header_has_numeric(header):
    pattern = r'^[-+]?\d*\.?\d+$'
    for column in header:
        if re.match(pattern, column):
            return True
    return False

def detect_line_delimiter(filename):
    with open(filename, "r", newline="") as file:
        ts_delimiter = csv.Sniffer().sniff(file.readline()).delimiter

    return ts_delimiter

def load_ts_from_csv(filename, has_header=None):
    ts_delimiter = detect_line_delimiter(filename)

    header = csv_has_header(filename, ts_delimiter, has_header)
    skiprows = 1 if has_header else 0

    return np.loadtxt(filename, delimiter=ts_delimiter, skiprows=skiprows), header


def load_ts_from_path(path, header_ts1, has_header=None):
    time_series = {}
    if os.path.isfile(path):
        ts2, header_ts2 = load_ts_from_csv(path, has_header)
        # TODO: Hacer un metodo que sea check_headers(header_ts1, header_ts2) que compruebe que los headers son iguales (elimina el codigo duplicado)
        if (header_ts1 != header_ts2) and has_header is not None:
            raise ValueError("All time series must have the same header column names.")
        time_series[os.path.basename(path)] = ts2
    else:    
        for _, _, files in os.walk(path):
            for file in files:
                ts2, header_ts2 = load_ts_from_csv(f"{path}/{file}", has_header)
                if (header_ts1 != header_ts2) and has_header is not None:
                    raise ValueError("All time series must have the same header column names.")
                time_series[file] = ts2

    return time_series
