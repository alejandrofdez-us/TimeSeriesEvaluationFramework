from SimilarityAnalysisComputer import SimilarityAnalysisComputer


class PlotComputer(SimilarityAnalysisComputer):

    def compute_next_analysis(self):
        plot = next(self.analysis_iterator)
        filename, ts_dict = self.current_associated_window
        return filename, plot.get_name(), plot.compute(self.core, filename)
