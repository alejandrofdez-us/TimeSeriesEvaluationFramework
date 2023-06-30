
[![version][contributors-shield]][contributors-url]
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

[contributors-shield]: https://img.shields.io/badge/last_update-07/XY/2023-brightgreen
[contributors-url]: https://img.shields.io/badge/last_update-07/XY/2023-brightgreen

![last-update](https://img.shields.io/badge/last_update-07/XY/2023-brightgreen)
![python](https://img.shields.io/badge/python-3.9)
![license](https://img.shields.io/badge/license-Apache_2.0-brightgreen)

# Project Name

A brief description of your project goes here.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

Provide a detailed overview of your project. Explain what it does, why it is useful, and any other relevant information. This section should give users a clear understanding of the purpose and scope of your project.

## Installation

Describe how to install and set up your project. Include any prerequisites or dependencies that need to be installed, as well as step-by-step instructions to get your project up and running.

```shell
$ git clone https://github.com/your-username/your-repository.git
$ cd your-repository
$ npm install  # or any other necessary commands
```

## Usage

## Contributing

## License




# TimeSeriesEvaluationFramework
Usage example everything:
```Bash
python evaluation_metrics.py --experiment_dir experiments/demoazure-epochs --trace azure_v2  --stride_ori_data_windows 12 --metrics all --recursive true --generate_deltas true --recompute_metrics true --inter_experiment_figures true --only_best_samples_figures 2
```
Usage example no deltas:
```Bash
python evaluation_metrics.py --experiment_dir experiments/demoazure-epochs --trace azure_v2  --stride_ori_data_windows 12 --metrics all --recursive true --generate_deltas false --recompute_metrics true --inter_experiment_figures true --only_best_samples_figures 2
```
Usage example no sdv:
```Bash
python evaluation_metrics.py --experiment_dir experiments/demoazure-epochs --trace azure_v2  --stride_ori_data_windows 12 --metrics all-no-sdv --recursive true --generate_deltas false --recompute_metrics true --inter_experiment_figures true --only_best_samples_figures 2
```

Usage example no sdv no tsne no pca:
```Bash
python evaluation_metrics.py --experiment_dir experiments/demoazure-epochs --trace azure_v2  --stride_ori_data_windows 12 --metrics all-no-sdv-no-tsne-pca --recursive true --generate_deltas false --recompute_metrics true --inter_experiment_figures true --only_best_samples_figures 2
```