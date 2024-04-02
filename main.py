import numpy as np
from Activations import Activation
import time
from numpy import ndarray


class Perceptron:
    def __init__(self, actvation, n_weights, weights=None):
        self.n_weights = n_weights + 1 

        if weights is None:
            self.weights = (2*np.random.rand(n_weights+1))-1  # gera valores de parametros entre -1 e 1
        else:
            self.weights = weights
        
        match actvation:
            case 'relu':
                self.activation = Activation.relu

            case 'sigmoid':
                self.activation = Activation.sigmoid

            case 'tanh':
                self.activation = Activation.tanh
            
            case _:
                pass

    def process_input(self, inputs: ndarray) -> float:
        sum = np.dot(self.weights, inputs)
       
        return self.activation(sum)


class MLP:
    def __init__(self, n_inputs, n_hiden_layer, n_exits, activation='relu'):
        self.inputs = []
        self.exits = []
        self.hidden = []
        self.bias_value = 1
        self.count = 0
        self.network = [self.inputs, self.hidden, self.exits]
        
        for i in range(n_inputs + 1):
            self.inputs.append(Perceptron(activation, 0))

        for i in range(n_hiden_layer + 1):
            self.hidden.append(Perceptron(activation, n_inputs))

        for i in range(n_exits + 1):
            self.exits.append(Perceptron(activation, n_hiden_layer))

    def foward(self, input_data):
        
        input_data.append(self.bias_value)
    
        output_layer1 = np.array(input_data)
        output_layer2 = []
        output_layer3 = []
        
        start = time.time()
        for neuron in self.hidden:
            output_layer2.append(neuron.process_input(output_layer1))

        output_layer2 = np.array(output_layer2)

        for neuron in self.exits:
            output_layer3.append(neuron.process_input(output_layer2))

        end = time.time()
        duracao = end - start
        print(duracao)
            
        return output_layer3


if __name__ == '__main__':
    values = [5, 3, 4, 2, 8, 6]
    nn = MLP(6, 1000, 3)
    print(nn.foward(values))
