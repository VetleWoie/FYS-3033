from basic_layers import Layer
import numpy as np

class Conv2D(Layer):
    def __init__(self, kernel_size,stride,num_kernels=1, learning_rate=0.1, padding = True, padding_type='zeros') -> None:
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_kernels=num_kernels
        self.learning_rate=learning_rate
        self.padding = padding
        self.padding_type = padding
        self.kernels = []

        for i in range(self.num_kernels):
            self.kernels.append(np.random.default_rng().uniform(low=-1, high=1,size=self.kernel_size))

    def _init_weights(self, prev_outputs:int):
        if self.padding:
            #If padding is enabled then output shape will be equall to input shape
            self.output_shape = (prev_outputs[0],prev_outputs[1],self.num_kernels)
        else:
            #If padding is disabled then output will be less than input
            self.output_shape = (prev_outputs.shape[0]-self.kernel_size[0]//2, prev_outputs.shape[1]-self.kernel_size[1]//2, self.num_kernels)

    def evaluate(self, prev_output):
        raise NotImplementedError
    
    def calculate_error(self, label=None,cost=None,next_error = None, next_weights = None):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError

if __name__=="__main__":
    l = Conv2D((3,3),1, num_kernels=3)
    for k in l.kernels:
        print(k)