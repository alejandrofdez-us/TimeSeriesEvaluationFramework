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