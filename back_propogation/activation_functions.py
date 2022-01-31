import numpy as np

class Step():
    def __call__(self, x) -> int:
        return -1 if x < 0 else 1

class Logistic():
    def __init__(self, a=1) -> None:
        self.a = a
    def __call__(self, x):
        return 1/(1+np.exp(-self.a*x))
    def derivative(self, x) -> float:
        return self.a*self(x) * (1-self(x))