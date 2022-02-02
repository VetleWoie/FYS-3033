from cost_functions import Quadratic_cost

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



class Neural_Net():
    def __init__(self,input_shape,layers, cost = Quadratic_cost()) -> None:
        self.layers = layers
        self.cost = cost

        self.layers[0]._init_weights(input_shape)
        for prevlayer,layer in zip(self.layers[:-1],self.layers[1:]):
            layer._init_weights(prevlayer.output_shape)

    def evaluate(self, datapoint):
        """
        Forward propagate the datapoint through all layers in the network

        Paramaters:
        datapoint: numpy.ndarray
        """
        #Evaluate input on first layer
        out = self.layers[0].evaluate(datapoint)
        #Forward propagate output through the rest of the layers
        for layer in self.layers[1:]:
            out = layer.evaluate(out)
        return out

    def back_propagate(self,label) -> None:
        """
        Back propagate error through network layers

        Paramaters:
        label: np.ndarray with training labels
        """
        self.layers[-1].calculate_error(label=label,cost=self.cost)
        for layer,nextlayer in zip(reversed(self.layers[:-1]),reversed(self.layers[1:])):
            layer.calculate_error(next_error = nextlayer.error, next_weights = nextlayer.weights)

    def update_weights(self) -> None:
        """
        Update weights in the network based on back propogation.
        """
        for layer in self.layers:
            layer.update_weights()

    def print_errors(self):
        for i,layer in enumerate(self.layers):
            print(f"Error in layer {i}:")
            print(layer.error)

    def __str__(self) -> str:
        string = ""
        for layer in self.layers:
            string += str(layer)
        return string