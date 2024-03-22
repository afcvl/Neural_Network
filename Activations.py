import numpy as np
from numba import jit ,types,njit
from numba import njit
class Activation():
    @staticmethod
    @jit(nopython=True)
    def ReLU(value):
        return max(0, value)

    @staticmethod
    @jit(nopython=True)
    def tanh (value):
        return np.tanh(value)

    @staticmethod
    @njit()
    def sigmoid( value):
        return 1 / (1 + np.exp(-value))