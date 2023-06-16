#Propiedades y métodos de esta clase:

class Metric:
    def __init__(self):
        pass

    def compute (self, ts1, ts2):
       raise NotImplementedError('Subclasses must implement compute() method')
