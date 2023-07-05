[![version](https://img.shields.io/badge/version-2.0-blue)](https://github.com/alejandrofdez-us/TimeSeriesEvaluationFramework/releases)
[![Python 3.9](https://img.shields.io/badge/python-3.9-darkgreen)](https://www.python.org/downloads/release/python-390/)
[![last-update](https://img.shields.io/badge/last_update-07/XY/2023-brightgreen)](https://github.com/alejandrofdez-us/TimeSeriesEvaluationFramework/commits/main)
![license](https://img.shields.io/badge/license-MIT-orange)

# Time Series Evaluation Framework

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Description

Time Series Evaluation Framework is an open-source project designed to facilitate the evaluation and comparison of
multivariate time series data. It provides a comprehensive toolkit for analyzing, visualizing, and reporting multiple
metrics and figures derived from time series datasets. The framework simplifies the process of evaluating time series by
offering data preprocessing, metrics computation, visualization, statistical analysis, and report generation
functionalities. With its customizable features, Time Series Evaluation Framework empowers researchers and data
scientists to gain insights, identify patterns, and make informed decisions based on their time series data.

### Available metrics

This toolkit can compute the following metrics:

- `kl`: Kullback-Leibler divergence
- `js`: Jensen-Shannon divergence
- `ks`: Kolmogorov-Smirnov test
- `mmd`: Maximum Mean Discrepancy
- `dtw` Dynamic Time Warping
- `cc`: Difference of co-variances
- `cp`: Difference of correlations
- `hi`: Difference of histograms

### Available figures

This toolkit can generate the following figures:

- `2D`: the ordinary graphical representation of the time series in a 2D figure with the time represented on the x axis
  and the data values on the y-axis for
    - the complete multivariate time series; and
    - a plot per column.

  Each generated figure plots both the original and the synthetically generated data to easily obtain key insights into
  the similarities or differences between them.
- `deltas`: the differences between the values of each column grouped by periods of time. For instance, the differences
  between the cpu usage every 5 minutes or every 30 minutes. These deltas can be used as a means of comparison between
  time series short-/mid-/long-term patterns.
  ![Delta Image grouped by 2 minutes](docs/images/mini_sample_1/deltas/cpu_TS_1_vs_TS_2_(grouped_by_2_minutes).png)
  ![Delta Image grouped by 5 minutes](docs/images/mini_sample_1/deltas/cpu_TS_1_vs_TS_2_(grouped_by_5_minutes).png)
  ![Delta Image grouped by 10 minutes](docs/images/mini_sample_1/deltas/cpu_TS_1_vs_TS_2_(grouped_by_10_minutes).png)
- `pca`: the linear dimensionality reduction technique that aims to find the principal components of a data set by
  computing the linear combinations of the original characteristics that explain the most variance in the data.
  ![PCA Image](docs/images/pca/PCA.png)
- `tsne`: a tool for visualising high-dimensional data sets in a 2D or 3D graphical representation allowing the creation
  of a single map that reveals the structure of the data at many different scales.
  ![TSNE Image 300 iterations 40 perplexity](docs/images/tsne/t_SNE_iter_300-perplexity_40.png)
  ![TSNE Image 1000 iterations 40 perplexity](docs/images/tsne/t_SNE_iter_1000-perplexity_40.png)
- `dtw` path: In addition to the numerical similarity measure, the graphical representation of the DTW path of each
  column can be useful to better analyse the similarities or differences between the time series columns. Notice that
  there is no multivariate representation of DTW paths, only single column representations.
  ![DTW Image for cpu](docs/images/mini_sample_1/dtw/DTW_cpu.png)

## Installation

To get started, follow these steps to install the Time Series Evaluation Framework:

### Step 1. Clone the repository to your local machine:

```Bash
git clone https://github.com/alejandrofdez-us/TimeSeriesEvaluationFramework
```

### Step 2. Navigate to the project directory:

```Bash
cd TimeSeriesEvaluationFramework
```

### Step 3. Install the required dependencies:

```Bash
pip install -r requirements.txt
```

## Usage

Users must provide `.csv` files containing multivariate time series by using the arguments `-ts1` and `-ts2_path`. Those
time series should not include a timestamp column and if a header is present, use the `-head` argument. The column
delimiter is automatically detected.

- `-ts1` should point to a single `csv` filename. This time series may represent the baseline or ground truth time
  series.
- `-ts2_path` can point to another single `csv` filename or a directory that contains multiple `csv` files to be
  compared with `-ts1` file.
- `-head` if your time series files include a header this argument must be present. If not present, the software
  understands that csv files don't include a header row.

Constraints:

- `-ts1` time-series file and `-ts2_path` time-series file(s) must have the same dimensionality (number of columns).
- all `-ts2_path` time-series files must have the same length (number of rows).
- `-ts1` may be a longer time series (more rows) than those included in `-ts2_path` files.

If `-ts1` time-series file is longer (more rows) than `-ts2_path` time-series file(s), the `-ts1` time series will be
divided in windows of the same
length as the `-ts2_path` time-series file(s).

For each `-ts2_path` time-series file, the most similar window (*) from `-ts1` time series is selected.

Finally, metrics and figures that assess the similarity between each pair of `-ts2_path` file
time-series and its associated most similar `-ts1` window are computed.

(*) `-w_select_met` is the metric used for the selection of the most
similar `-ts1` time-series window per each `--ts2_path` time-series file(s).`dtw` is the default value, however, any of
the
[metrics](#available-metrics) are also available for this purpose using this argument.

Users must provide at least a metric or a figure to be computed/generated:

- `-m` the [metrics](#available-metrics) names to be computed as a list separated by spaces.
- `-f` the [figures](#available-figures) names to be computed as a list separated by spaces

The following arguments are also available for fine-tuning:

- `-ts_freq_secs` the frequency in seconds in which samples were taken just to generate the delta figures. By default is
  one second.
- `-strd` when `ts1` time-series is longer than `ts2_path` time-series file(s) the windows are computed by using a
  stride of 1 by default. Sometimes using a larger value for the stride parameter improves the performance by skipping
  the computation of similarity between so many windows.

### Basic usage examples:

The following examples of evaluation of similarity are shown below:

1. Two time series computing only DTW metric and DTW figure:
    ```Bash
    python main.py -ts1 data/example_1.csv --ts2_path experiments/mini_example_1.csv --metrics dtw --figures dtw
    ```

1. A time series and all time series within a directory computing only DTW metric and DTW figure::

    ```Bash
    python main.py -ts1 data/example_1.csv -ts2_path experiments -m dtw -f dtw
    ```

1. A time series and all time series within a directory computing every metric and figure available:

    ```Bash
    python main.py -ts1 data/example_1.csv -ts2 experiments -m cc cp dtw hi js kl ks mmd -f deltas dtw evolution pca tsne
    ```

1. Comparison using filenames whose first rows are used as headers (all filenames must contain the same header):

    ```Bash
    python main.py -ts1 data/example_1.csv -ts2 experiments -m dtw -f dtw -head
    ```

1. Comparison between time series specifying the frequency in seconds in which samples were taken:

    ```Bash
    python main.py -ts1 data/example_1.csv -ts2_path experiments -m dtw -f dtw --timestamp_frequency_seconds 60
    ```

1. Comparison between time series specifying the stride that determines the step or distance by which a fixed-size
   window
   moves over the first time series:

    ```Bash
    python main.py -ts1 data/example_1.csv -ts2_path experiments -m dtw -f dtw -strd 5
    ```

1. Comparison between time series specifying the window selection metric to be used when selecting the best windows in
   the first time series:

    ```Bash
    python main.py -ts1 data/example_1.csv -ts2_path experiments -m dtw -f dtw --window_selection_metric js
    ```

1. Using our sample time series to compute every single metric and figure:

    ```Bash
    python main.py -ts1 data/sample_1.csv -ts2_path experiments -head -m cc cp dtw hi js kl ks mmd -f deltas dtw evolution pca tsne -w_select_met cc -ts_freq_secs 60 -strd 5
    ```

Every metric computation will be found in the `results` directory and every figure generated will be found at `figures`
directory

## Advanced usage

Additionally, users may implement their own metric or figure classes an include them within the `metrics` or `plots`
directory. To ensure compatibility with our framework, they have to inherit from the base classes (`Metric` and `Plot`)
and include their implemented classes as options in the argument parser found in `main.py`.

This allows the framework to dynamically recognize and utilize these custom classes based on user input. By including
them in the argument parser, users can easily select their custom metrics or plots when running the framework, ensuring
that their classes are properly integrated and applied during the time series evaluation process.

## License

Time Series Evaluation Framework is free and open-source software licensed under the [MIT license](LICENSE).