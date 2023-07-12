class PlotComputer:

    def __init__(self, core, ts1_ts2_associated_windows, plots):
        self.ts1_ts2_associated_windows_iterator = iter(ts1_ts2_associated_windows.items())
        self.core = core
        self.plots = plots
        self.plots_iterator = iter(plots)
        self.length = len(plots) * len(
            ts1_ts2_associated_windows.items())  # FIXME: arreglar, tsne y pca no se repite para cada fichero
        self.already_computed_figures_requires_all_samples = []

    def __iter__(self):
        self.current_associated_window = next(self.ts1_ts2_associated_windows_iterator)
        return self

    def __next__(self):
        try:
            return self.__generate_next_plot()
        except StopIteration:
            self.already_computed_figures_requires_all_samples = []
            self.current_associated_window = next(self.ts1_ts2_associated_windows_iterator)
            self.plots_iterator = iter(self.plots)
            return self.__generate_next_plot()

    def __len__(self):
        return self.length

    def __generate_next_plot(self):
        plot = next(self.plots_iterator)
        filename, ts_dict = self.current_associated_window
        if plot.get_name() not in self.already_computed_figures_requires_all_samples:
            generated_plots = plot.generate_figures(self.core, filename)
            if plot.requires_all_samples():
                self.already_computed_figures_requires_all_samples.append(plot.get_name())
        return filename, plot.get_name(), generated_plots
