class PlotConfig:
    def __init__(self, figures=None, timestamp_frequency_seconds=1):
        if figures is None:
            figures = ['2d', 'deltas', 'dtw', 'pca', 'tsne']
        self.figures = figures
        self.timestamp_frequency_seconds = timestamp_frequency_seconds
