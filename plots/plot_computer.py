import warnings
from similarity_analysis_computer import SimilarityAnalysisComputer


class PlotComputer(SimilarityAnalysisComputer):

    def __init__(self, core, analysis):
        super().__init__(core, analysis)
        self.already_computed_figures_requires_all_samples = []

    def _compute_next_analysis(self):
        plot = next(self.analysis_iterator)
        ts2_filename, _ = self.current_associated_window
        computed_plots = []
        if plot.get_name() not in self.already_computed_figures_requires_all_samples:
            try:
                if plot.requires_all_samples():
                    self.already_computed_figures_requires_all_samples.append(plot.get_name())
                computed_plots = plot.compute(self.core, ts2_filename)
            except Exception as e:
                warnings.warn(f'\nWarning: Plot {plot.get_name()} could not be computed. Details: {e}', Warning)
        return ts2_filename, plot.get_name(), computed_plots
