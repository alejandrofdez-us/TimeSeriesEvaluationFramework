# TimeSeriesEvaluationFramework
Sample run:
```Bash
python evaluation_metrics.py --ori_data_filename data/trazas_google/week1/instance_usage_5min_sample_week1.csv --experiment_dir experiments/demoroot/  --metrics mmd,dtw,kl,cc,cp,hi,evolution_figures --trace google2019 --recursive true
```