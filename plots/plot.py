from matplotlib import pyplot as plt


class Plot:
    @staticmethod
    def requires_all_samples():
        return False

    def __init__(self, fig_size=(18, 3)):
        self.filename = None
        self.ts1 = None
        self.ts2 = None
        self.ts1_windows = None
        self.header_names = None
        self.fig_size = fig_size

    def initialize(self, core, filename):
        self.filename = filename
        self.ts1 = core.ts1_ts2_associated_windows[self.filename]["most_similar_ts1_sample"]
        self.ts2 = core.ts1_ts2_associated_windows[self.filename]["ts2"]
        self.ts1_windows = core.ts1_windows
        self.header_names = core.header_names

    def get_name(self):
        return self.__class__.__name__.lower()

    def compute(self, core, filename):
        self.initialize(core, filename)
