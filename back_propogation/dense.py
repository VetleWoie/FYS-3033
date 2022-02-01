import numpy as np
from activation_functions import Logistic

class Dense():
    def __init__(self, nodes, learning_rate = 0.1, activation = Logistic(), weights = None, bias = None):
        """
        Nodes: Number of neurons in the layer
        Activation: Instance of an activation function class:
                    Needs to have defined both a __call__ method and
                    a derivative method
        """
        self.nodes = nodes
        self.activation = activation
        self.learning_rate = learning_rate
        #Initialize bias values

        self.counter = 0
        self.bias = bias
        if self.bias is None:
            self.bias = np.random.rand(self.nodes,1)

        self.weights = weights
    
    def _init_weights(self, prev_outputs:int):
        """
        Initialize weights based on the number of outputs in the previous layer
        
        Paramaters:
        prev_outputs: int
        """
        self.inputs = prev_outputs
        if self.weights is None:
            self.weights = np.random.default_rng().uniform(low=-1, high=1,size=(self.nodes, prev_outputs))
    
    def classify(self, prev_output):
        """
        Classify point based on previous output

        Paramaters:
        prev_output: numpy.ndarray
        """
        self.input = prev_output
        self.v = self.weights @ prev_output + self.bias
        self.y = self.activation(self.v)
        return self.y
    
    def calculate_error(self, label=None,cost=None,next_error = None, next_weights = None):
        """
        Calculate error in current layer based on either output cost or the error in the 
        next layer.

        Paramaters:
            label: Known class of input datapoint
            cost: Cost function used for measuring cost of network
            next_error: Error in next layer
            next_weights: Weights from next layer
        """
        if next_error is None and next_weights is None and cost is not None and label is not None:
            #If layer is output layer calculate cost as output cost
            self.error = cost.derivative(self.y, label) * self.activation.derivative(self.v)
        elif next_error is not None and next_weights is not None:
            #If layer is not output layer calculate error using the error in the next layer
            self.error = next_weights.T @ next_error * self.activation.derivative(self.v)
        elif cost is None or label is None:
            #If missing cost function or label when calculating output error then raise an exeption
            raise ValueError("Need cost function and label to calculate output error")
        else:
            #If missing weights or error from next layer when calculating error then raise an exeption
            raise ValueError("Need both weights and error from the next layer to calculate error")
        
        #Calculate update delta
        self.delta_weights = self.error @ self.input.T
        self.delta_bias = np.sum(self.error,axis=1,keepdims=True)
        #Keep track of how many errors that have been calculated since last update
        return self.error

    def update_weights(self):
        #Update weights using stocastic gradient descent
        self.weights = self.weights - (self.learning_rate/4) * self.delta_weights
        self.bias = self.bias -(self.learning_rate/4) * self.delta_bias

    def __str__(self):
        return f"Weights: {self.weights.shape}\n{self.weights}\nBias:\n{self.bias}\n"
 