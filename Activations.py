import numpy as np
from numba import jit, njit


class Activation(object):
    @staticmethod
    @njit()
    def relu(value: float) -> float:
        return np.maximum(0, value)

    @staticmethod
    @njit()
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    @staticmethod
    @njit()
    def tanh(value):
        return np.tanh(value)

    @staticmethod
    @jit(nopython=True)
    def linear(value):
        return value



if __name__ == "__main__":
    print(Activation.relu(1.0))
