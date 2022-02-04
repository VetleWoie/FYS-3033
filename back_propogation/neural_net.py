from cost.cost_functions import Quadratic_cost
from progress_bar import progress_bar

class Neural_Net():
    def __init__(self,input_shape,layers, cost = Quadratic_cost()) -> None:
        self.layers = layers
        self.cost = cost

        self.layers[0]._init_weights(input_shape)
        for prevlayer,layer in zip(self.layers[:-1],self.layers[1:]):
            layer._init_weights(prevlayer.output_shape)
    
    def fit(self,epochs, batch_size, training, testing=None):
        if testing is not None:
            test_set = testing[0]
            test_labels = testing[1]
        training_set = training[0]
        training_labels = training[1]

        test_cost = []
        training_cost = []

        for epoch in range(epochs):
            print("Epoch: ", epoch)
            progress_bar(epoch, epochs, show_total=True, show_percentage=False)
            correct = 0
            training_out = self.evaluate(training_set)

            training_cost.append(self.cost(training_out, training_labels)[0,0]/len(training_set))
            if testing is not None:
                test_out = self.evaluate(test_set)
                test_cost.append(self.cost(test_out, test_labels)[0,0]/len(test_set))
                
            for i in range(0,len(training_set),batch_size):
                print(i)
                start = i
                stop = start+batch_size
                batch = training_set[start:stop]
                batch_label = training_labels[start:stop]
                out = self.evaluate(batch)
                self.back_propagate(batch_label)
                self.update_weights()

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