import numpy as np
from Activations import Activation
import time
from numpy import ndarray


class Perceptron:
    def __init__(self, actvation, n_weights, weights=None):
        self.n_weights = n_weights + 1
        self.count = 0

        if weights is None:
            self.weights = (2 * np.random.rand(n_weights + 1).astype(np.float64)) - 1
            # gera valores de parametros entre -1 e 1
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
        soma = np.dot(self.weights, inputs)
        
        return self.activation(soma)


class MLP:
    def __init__(self, size_layers, activation='relu'):
        self.network = []
        
        n_weights = 0
        
        for size in size_layers:
            
            neurons_list = []
            
            for _ in range(size):
                neurons_list.append(Perceptron(activation, n_weights=n_weights))

            self.network.append(neurons_list)
            
            n_weights = size

    def foward(self, input_data):    
        input_values = input_data.copy()  # Cria uma cópia da lista original
        input_values.append(1)
        input_values = np.array(input_values, dtype=np.float64)
        output_values = []

        for layer in self.network[1:]:  # itera sobre as camadas excluindo a de entrada

            for neuron in layer:
                output_values.append(neuron.process_input(input_values))
            output_values.append(1)  # append bias

            input_values = np.array(output_values, dtype=np.float64) # seta input_values que será usado na proxima camada
            output_values = []

        output_values = np.delete(input_values, -1)

        return output_values


if __name__ == '__main__':
    values = [5, 3, 4, 2, 8, 6, 5, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4]
    
    nn = MLP(size_layers=[len(values), 20000, 50, 1000, 10])
    
    
    n_runs = 10
    start = time.time()
    
    for i in range(n_runs):
        nn_output =  nn.foward(values)
    
    end = time.time()
    avarege_time = (end - start)/n_runs
    
    print("saida da rede:", nn.foward(values).tolist())
    print("tempo medio:", avarege_time, "s")