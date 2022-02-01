from neural_net import Layer

class convolution(Layer):
    def __init__(self, kernel,stride=1) -> None:
        pass

    def _init_weights(self, prev_outputs:int):
        raise NotImplementedError

    def evaluate(self, prev_output):
        raise NotImplementedError
    
    def calculate_error(self, label=None,cost=None,next_error = None, next_weights = None):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError