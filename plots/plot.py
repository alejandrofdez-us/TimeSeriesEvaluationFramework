from matplotlib import pyplot as plt


class Plot:
    @staticmethod
    def requires_all_samples():
        return False

    def __init__(self, plot_size=(18, 3)):
        self.filename = None
        self.ts1 = None
        self.ts2 = None
        self.ts1_windows = None
        self.header_names = None
        plt.rcParams["figure.figsize"] = plot_size

    def initialize(self, core, filename):
        self.filename = filename
        self.ts1 = core.ts1_ts2_associated_windows[self.filename]["most_similar_ts1_sample"]
        self.ts2 = core.ts1_ts2_associated_windows[self.filename]["ts2"]
        self.ts1_windows = core.ts1_windows
        self.header_names = core.header_names

    def get_name(self):
        return self.__class__.__name__.lower()

    def generate_figures(self, core,
                         filename):  # FIXME: renombrar a generate o generate_plots o compute si lo igualamos a metric
        self.initialize(core, filename)
