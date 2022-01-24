import numpy as np
from matplotlib import pyplot as plt

class Logistic():
    def __call__(self, x,a=10):
        return 1/(1+np.exp(-a*x))
    def derivative(self, x,a=10) -> float:
        return self(x) - (1-self(x))

class Quadratic_cost():
    def __call__(self, label, output) -> float:
        return np.dot(label-output,label-output)
    def derivative(self, label, output):
        return (output-label)

class Dense():
    def __init__(self, nodes, activation = Logistic()):
        """
        Nodes: Number of neurons in the layer
        Activation: Instance of an activation function class:
                    Needs to have defined both a __call__ method and
                    a derivative method
        """
        self.nodes = nodes
        self.activation = activation
        #Initialize bias values
        self.bias = np.random.rand(1,self.nodes)
    
    def _init_weights(self, prev_nodes):
        self.weights = np.random.rand(self.nodes, prev_nodes)
    
    def classify(self, datapoint):
        z = self.weights @ datapoint + self.bias

    
class NeuralNet():

    def __init__(self, layers, activation = Logistic(), cost = Quadratic_cost()) -> None:
        self.layers = layers
        self.weights = [(np.random.rand(self.layers[i], self.layers[i-1])) for i in range(1,len(self.layers))]
        self.bias = [np.random.rand(1,self.layers[d]) for d in self.layers]

        self.activation = activation
        self.cost = cost
    
    def classify(self, p) -> np.array:
        self.z = []
        self.a = []
        self.a.append(p)
        for i,(w,b) in enumerate(zip(self.weights,self.bias), start=1):
            z = w @ self.a[i-1] + b
            self.z.append(z)
            self.a.append(self.activation(z))
        return self.a[-1]

    def output_error(self, label):
        return self.cost.derivative(label, self.a[-1]) * self.activation.derivative(self.z[-1])

    def back_propagate(self, label):
        error = []
        print("Label: ", label)
        print(f"Num weights: {len(self.weights)}, num z: {len(self.z)}")

        print("Z: ")
        for z in self.z:
            print(z)
            print(",")
        print()

        print("A: ")
        for a in self.a:
            print(a.shape)
            print(a)
            print(",")
        print()

        print("w: ")
        for w in reversed(self.weights):
            print(w)
            print(w.shape)
            print(",")
        print()

        error.append(self.output_error(label))
        print(f"Error: {error[0]}")
        for i,(w,z) in enumerate(zip(reversed(self.weights), reversed(self.z[:-1]))):
            print(i,w.shape,z.shape, error[i].shape)
            delta = w.T @ error[i] * self.activation.derivative(z)
            error.append(delta)
        
        print()
        print("Error:")
        for e in error:
            print(e.shape)
            print(e)
            print(",")

        print(error[1].shape)
        print(self.a[1].shape)
        dcdw1 = np.outer(self.a[1], error[1])
        print(dcdw1)
        dcdw0  = np.outer(self.a[2], error[0])
        print(dcdw0)
        
    def train(self, training_set, training_label,epoch, batch_size, learning_rate) -> None:
        #Loop over all epochs
        for e in range(epoch):
            cost = 0
            num_batch = len(training_label) / batch_size
            #Train on a batch of datapoints
            for batch in range(num_batch):
                print(f"Epoch: {e} Batch: {batch} ", end="")
                #Loop over datapoints in batch
                for i in range(batch_size):
                    datapoint = training_set[i+e*i]
                    label = training_label[i+e*i]
                    #Classify datapoint and compute cost
                    cost += self.cost_function(self.classify(datapoint))
                    #Propagate error backwards in network
                    self.back_propagate(datapoint, label)
                #Calculate avarage cost
                cost *= 1/(2*batch_size)
                print(f"Cost = {cost}")

if __name__ == "__main__":
    np.set_printoptions(precision=2)

    trainingSet = [
        np.array([[0],[0]]),
        np.array([[1],[0]]),
        np.array([[0],[1]]),
        np.array([[1],[1]]),
    ]

    trainingLabel = [-1,1,1,-1] 

    nn = NeuralNet([2,2,1])

    datapoint = trainingSet[0]
    label = trainingLabel[0]

    out = nn.classify(datapoint)
    output_error = nn.output_error(label)
    bp = nn.back_propagate(label)


    # print(f"Weights: {nn.weights[-1]}")


    # print(f"Input: {datapoint}, Label: {label}")
    # print(f"Output: {out}")
    # print(f"Output Error: {output_error}")
    # print(f"Back Prop result: {bp}")
