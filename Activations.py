import numpy as np
from numba import jit, njit


class Activation(object):
    @staticmethod
    @njit()
    def relu(value: float) -> float:
        return np.maximum(0, value)
    
    @staticmethod
   # @njit()
    def d_relu(x):
        return (x > 0).astype(float)

    @staticmethod
    @njit()
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))
    
    @staticmethod
    def d_sigmoid(value):
        return Activation.sigmoid(value) *  (1 - Activation.sigmoid(value))

    @staticmethod
    @njit()
    def tanh(value):
        return np.tanh(value)
    
    @staticmethod
    def d_tanh(value):
        return 1 - np.tanh(value)**2

    @staticmethod
    @jit(nopython=True)
    def linear(value):
        return value



if __name__ == "__main__":
    print(Activation.relu(1.0))
