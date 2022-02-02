class Layer():
    def __init__(self, shape) -> None:
        self.output_shape = shape

    def _init_weights(self, prev_outputs:int):
        raise NotImplementedError

    def evaluate(self, prev_output):
        raise NotImplementedError
    
    def calculate_error(self, label=None,cost=None,next_error = None, next_weights = None):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError
 

class Input(Layer):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def _init_weights(self, prev_outputs:int):
        pass

    def evaluate(self,prev_output):
        return prev_output
    
    def calculate_error(self, label=None,cost=None,next_error = None, next_weights = None):
        pass

    def update_weights(self):
        pass