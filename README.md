[![version](https://img.shields.io/badge/version-2.0-blue)](https://github.com/alejandrofdez-us/TimeSeriesEvaluationFramework/releases)
[![Python 3.9](https://img.shields.io/badge/python-3.9-darkgreen)](https://www.python.org/downloads/release/python-390/)
[![last-update](https://img.shields.io/badge/last_update-07/XY/2023-brightgreen)](https://github.com/alejandrofdez-us/TimeSeriesEvaluationFramework/commits/main)
![license](https://img.shields.io/badge/license-MIT-orange)

# Time Series Evaluation Framework

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Time Series Evaluation Framework is an open-source project designed to facilitate the evaluation and comparison of multivariate time series data. It provides a comprehensive toolkit for analyzing, visualizing, and reporting multiple metrics and figures derived from time series datasets. The framework simplifies the process of evaluating time series by offering data preprocessing, metric calculation, visualization, statistical analysis, and report generation functionalities. With its customizable features, Time Series Evaluation Framework empowers researchers and data scientists to gain insights, identify patterns, and make informed decisions based on their time series data.

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
Users must provide `.csv` files containing multivariate simples. All time series included in the --time_series_2_path are expected to be the same size, only --time_series_1_filename may receive a time series equal or greater in size. If greater, the fist time series will be divided in windows in order to pick the best one for each time series given in --time_series_2_path. DTW is the default metric when it comes to picking this windows, any other metric is also available for this purpose using the --window_selection_metric argument.

Examples of usage are shown below: 

- Comparing two time series using DTW metric and figure parameters: 
```Bash
python main.py --time_series_1_filename data/example_1.csv --time_series_2_path experiments/mini_example_1.csv --metrics dtw --figures dtw
```

- Comparison between a time series and all time series within a directory:
```Bash
python main.py --time_series_1_filename data/example_1.csv --time_series_2_path experiments --metrics dtw --figures dtw
```

- Comparison using every metric and figure available:
```Bash
python main.py -ts1 data/example_1.csv -ts2 experiments --metrics cc cp dtw hi js kl ks mmd --figures deltas dtw evolution pca tsne
```

- Comparison using filenames whose first rows are used as headers (all filenames must contain the same header):
```Bash
python main.py -ts1 data/example_1.csv -ts2 experiments -m dtw -f dtw --header
```

- Comparison between time series specifying the frequency in seconds in which samples were taken:
```Bash
python main.py -ts1 data/example_1.csv -ts2 experiments -m dtw -f dtw --timestamp_frequency_seconds 60
```

- Comparison between time series specifying the stride to be used when selecting the best windows in the first time series:
```Bash
python main.py -ts1 data/example_1.csv -ts2 experiments -m dtw -f dtw --stride 5
```

- Comparison between time series specifying the window selection metric to be used when selecting the best windows in the first time series:
```Bash
python main.py -ts1 data/example_1.csv -ts2 experiments -m dtw -f dtw --window_selection_metric js
```

- Using our sample time series to compute every single metric and figure:
```Bash
python main.py -ts1 data/sample_1.csv -ts2 experiments -head -m cc cp dtw hi js kl ks mmd -f deltas dtw evolution pca tsne -w_select_met cc -ts_freq_secs 60 -strd 5
```

Every output will be found in the `results` directory.

Additionally, users may implement their own metric or figure classes an include them within the `metrics` or `plots` directory. To ensure compatibility with our framework, they have to inherit from the base classes (`Metric` and `Plot`) and include their implemented classes as options in the argument parser found in `main.py`. 

This allows the framework to dynamically recognize and utilize these custom classes based on user input. By including them in the argument parser, users can easily select their custom metrics or plots when running the framework, ensuring that their classes are properly integrated and applied during the time series evaluation process.

## License

Time Series Evaluation Framework is free and open-source software licensed under the [MIT license](<https://github.com>).