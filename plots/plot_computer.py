from SimilarityAnalysisComputer import SimilarityAnalysisComputer


class PlotComputer(SimilarityAnalysisComputer):

    def _compute_next_analysis(self):
        plot = next(self.analysis_iterator)
        filename, _ = self.current_associated_window
        try:
            computed_plots = plot.compute(self.core, filename)
        except Exception as e:
            computed_plots = []
            warnings.warn(f"\nWarning: Plot {plot.get_name()} could not be computed. Details: {e}", Warning)
        return filename, plot.get_name(), computed_plots
