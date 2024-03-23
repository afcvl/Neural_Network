import numpy as np
from numba import njit
from numba.typed import List
from Activations import Activation
import time
        
class Perceptron():
    def __init__(self, actvation,n_weights,weights=None):
        self.activation = None
        self.n_weights = n_weights

        if(weights is None):
            self.weights=(2*np.random.rand(n_weights))-1 #gera valores de parametros entre -1 e 1
        else:
            self.weights=weights
        
        match actvation:
            case 'relu':
                self.activation = Activation.ReLU

            case 'sigmoid':
                self.activation = Activation.sigmoid

            case 'tanh':
                self.activation = Activation.tanh
            
            case _:
                pass
    
    def process_input(self, inputs:np.ndarray) -> float:
        sum = np.dot(self.weights, inputs)
       
        return self.activation(sum)
    
class MLP():
    def __init__(self, n_inputs, n_hiden_layer, n_exits, activation='relu'):
        self.inputs = []
        self.exits = []
        self.hidden = []
        self.count = 0
        self.network = [self.inputs, self.hidden, self.exits]
        
        for i in range(n_inputs):
            self.inputs.append(Perceptron(activation, 0))

        for i in range(n_hiden_layer):
            self.hidden.append(Perceptron(activation, n_inputs))

        for i in range(n_exits):
            self.exits.append(Perceptron(activation, n_hiden_layer))

    #@jit(nopython=True)
    def foward(self, input_data):
        output_layer1 = np.array(input_data)
        output_layer2 = []
        output_layer3 = []
        
        start = time.time()
        for neuron in self.hidden:
            output_layer2.append(neuron.process_input(output_layer1))
        end = time.time()

        for neuron in self.exits:
            output_layer3.append(neuron.process_input(output_layer2))
            
        
        duracao = end - start
        
        print(duracao)
            
        return output_layer3

            

values = [5,3,4,2,8,6]

nn = MLP(n_inputs=6, n_hiden_layer=100, n_exits=5)

print(nn.foward(values))