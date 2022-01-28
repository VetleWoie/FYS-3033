import numpy as np
from matplotlib import pyplot as plt

class Step():
    def __call__(self, x) -> int:
        return -1 if x < 0 else 1

class Logistic():
    def __init__(self, a=1) -> None:
        self.a = a
    def __call__(self, x):
        return 1/(1+np.exp(-self.a*x))
    def derivative(self, x) -> float:
        return np.exp(-self.a*x)/(1+np.exp(-self.a*x))**2#self(x) - (1-self(x))

class Quadratic_cost():
    def __call__(self, output, label) -> float:
        return np.dot(label-output,label-output)
    def derivative(self, output, label):
        return (label-output)

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
        print("v:")
        print(self.v)
        self.y = self.activation(self.v)
        print("y:")
        print(self.y)
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
            print("cost derivative: ")
            print(cost.derivative(self.y, label))
            print("Activation derivative: ")
            print(self.activation.derivative(self.v))
            self.error = cost.derivative(self.y, label) * self.activation.derivative(self.v)
            # print("output error:", self.error)
        elif next_error is not None and next_weights is not None:
            #If layer is not output layer calculate error using the error in the next layer
            # print("w*e:")
            # print(next_weights.T @ next_error)
            # print("Activation derivative:")
            # print(self.activation.derivative(self.v))
            self.error = next_weights.T @ next_error * self.activation.derivative(self.v)
        elif cost is None or label is None:
            #If missing cost function or label when calculating output error then raise an exeption
            raise ValueError("Need cost function and label to calculate output error")
        else:
            #If missing weights or error from next layer when calculating error then raise an exeption
            raise ValueError("Need both weights and error from the next layer to calculate error")
        
        print("error:")
        print(self.error)
        #Calculate update delta
        self.delta_weights = self.error @ self.input.T
        self.delta_bias = np.sum(self.error)
        print("delta weight:")
        print(self.delta_weights)
        print("delta bias")
        print(self.delta_bias)
        #Keep track of how many errors that have been calculated since last update
        return self.error

    def update_weights(self):
        #Update weights using stocastic gradient descent
        self.weights = self.weights - (self.learning_rate/4) * self.delta_weights
        self.bias = self.bias -(self.learning_rate/4) * self.delta_bias

    def __str__(self):
        return f"Weights: {self.weights.shape}\n{self.weights}\nBias:\n{self.bias}\n"

class Neural_Net():
    def __init__(self,input_shape,layers, cost = Quadratic_cost()) -> None:
        self.layers = layers
        self.cost = cost

        self.layers[0]._init_weights(input_shape)
        for prevlayer,layer in zip(self.layers[:-1],self.layers[1:]):
            layer._init_weights(prevlayer.nodes)

    def evaluate(self, datapoint):
        """
        Forward propagate the datapoint through all layers in the network

        Paramaters:
        datapoint: numpy.ndarray
        """
        #Evaluate input on first layer
        out = self.layers[0].classify(datapoint)
        #Forward propagate output through the rest of the layers
        for layer in self.layers[1:]:
            out = layer.classify(out)
        return out

    def back_propagate(self,label) -> None:
        self.layers[-1].calculate_error(label=label,cost=self.cost)
        for layer,nextlayer in zip(reversed(self.layers[:-1]),reversed(self.layers[1:])):
            layer.calculate_error(next_error = nextlayer.error, next_weights = nextlayer.weights)

    def update_weights(self) -> None:
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

if __name__ == "__main__":
    np.random.seed(0)
    # np.set_printoptions(precision=2)

    trainingSet = [
        np.array([[0],[0]]),
        np.array([[1],[0]]),
        np.array([[0],[1]]),
        np.array([[1],[1]]),
    ]

    trainingLabel = [
        np.array([[0]]),
        np.array([[1]]),
        np.array([[1]]),
        np.array([[0]]),
    ]

    nn = Neural_Net(2,
        [
            Dense(2,learning_rate=1,weights=np.array([[0,1],[1,0]]), bias=np.array([[0], [0]])), 
            Dense(1,learning_rate=1,weights=np.array([[1,1]]),bias=np.array([[0]]))
        ])
    print(nn)

    print()
    print("Input:")
    p = np.array([[1,1,0,0],[1,0,1,0]])
    label = np.array([[0,1,1,0]])
    print(p.shape)
    print(p)
    for i in range(100):
        print()
        print("Evaluating:")
        out = nn.evaluate(p)
        print()
        print("Final output:")
        print(out)
        print()
        print("Propagating error backwards:")
        nn.back_propagate(label)
        print()
        print("Update weights")
        nn.update_weights()
        print("New network")
        print(nn)

    print(p[:,0].reshape(2,1))
    print("Evaluating:")
    out = nn.evaluate(p)
    print()
    print("Final output:")
    print(out)
