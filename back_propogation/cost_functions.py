import numpy as np

class Quadratic_cost():
    def __call__(self, output, label) -> float:
        return np.dot(label-output,label-output)
    def derivative(self, output, label):
        return (label-output)