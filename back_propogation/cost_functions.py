import numpy as np

class Quadratic_cost():
    def __call__(self, output, label) -> float:
        return np.dot((output-label),(output-label))
    def derivative(self, output, label):
        return (output-label)