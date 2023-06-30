class Plot:
    def __init__(self):
        pass

    def generate_figures (self, args):
       raise NotImplementedError('Subclasses must implement generate_figures() method')
    
    @staticmethod
    def requires_all_samples():
        return False
