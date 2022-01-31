import numpy as np
from dense import Dense
from cost_functions import Quadratic_cost
from activation_functions import Logistic, Step
from matplotlib import pyplot as plt

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
    for i in range(10000):
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
