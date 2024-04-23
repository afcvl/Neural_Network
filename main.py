import numpy as np
from Activations import Activation
from numpy import ndarray
import pickle
import time

def letra_para_codigo(letra):
    if 'A' <= letra <= 'Z':
        return ord(letra) - 64
    else:
        print("Valor não reconhecido.")
        return None

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def d_mse(y_true, y_pred):
    return 2 * (y_pred - y_true)

class Perceptron:
    def __init__(self, actvation, n_weights, weights=None):
        self.n_weights = n_weights + 1
        self.count = 0
        self.last_input = None
        self.soma = 0

        if weights is None:
            self.weights = (2 * np.random.rand(n_weights + 1).astype(np.float64)) - 1
            # gera valores de parametros entre -1 e 1
        else:
            self.weights = weights

        match actvation:
            case 'relu':
                self.activation = Activation.relu
                self.d_activation = Activation.d_relu

            case 'sigmoid':
                self.activation = Activation.sigmoid
                self.d_activation = Activation.d_sigmoid

            case 'tanh':
                self.activation = Activation.tanh

            case 'linear':
                self.activation = Activation.linear

            case _:
                pass

    def process_input(self, inputs: ndarray) -> float:
        
        self.last_input = np.array(inputs, dtype=np.float64)  # salva ultimo input
        self.soma = np.dot(self.weights, inputs)        # salva resultado da função de transição
        
        return self.activation(self.soma)
    
    
    def compute_gradient(self, d_error):
    
        return d_error * self.d_activation(self.soma) * self.last_input

class MLP:
    def __init__(self, layers_size):
        self.network = []

        for i in range(1, len(layers_size)):
            self.add_layer(n_inputs=layers_size[i-1], 
                           number_neurons=layers_size[i],
                           activation='sigmoid')

    def forward(self, input_data):
        input_values = input_data.copy()  # Cria uma cópia da lista original
        input_values.append(1)
        input_values = np.array(input_values, dtype=np.float64)

        for layer in self.network:  # itera sobre as camadas excluindo a de entrada
            layer_output_values = []

            for neuron in layer:
                layer_output_values.append(neuron.process_input(input_values))
            
            # seta input_values que será usado na proxima camada
            layer_output_values.append(1) # inserindo bias para proxima camada
            
            input_values = np.array(layer_output_values, dtype=np.float64)
  
        return layer_output_values[:-1] # Retira o bias da lista

    def add_layer(self, n_inputs, number_neurons, activation):
        neurons_list = []

        for _ in range(number_neurons):
            neurons_list.append(Perceptron(activation, n_inputs))

        self.network.append(neurons_list)

    def save_network(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.network, f)

    def load_network(self, file_name):
        with open(file_name, 'rb') as f:
            self.network = pickle.load(f)
   
    def train(self, inputs, expected_output, learning_rate):
        # feed forward
        outputs = self.forward(inputs)
        
       # print(f'Saída: {outputs}')
        
        outputs = np.array(outputs)

        error = d_mse(outputs, expected_output)

        # Backpropagation
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            next_error = np.zeros_like(layer[0].weights)

            for j, neuron in enumerate(layer):
                d_error = error[j]
                gradients = neuron.compute_gradient(d_error)
                
                # Atualização dos pesos
                neuron.weights += learning_rate * gradients

                # Preparar o erro para a próxima camada
                if i != 0:
                    next_error += neuron.weights * d_error

            error = next_error

        return outputs.tolist(), mse(expected_output, outputs[-1])
    
    
def prepara_features(file):
    inputs = []
    features_txt = open(file, 'r')
    for line in features_txt:
        line= line.split(',')
        line = [x.lstrip() for x in line]
        inputs.append(line[:-1])
        
    return inputs

def prepara_labels(file):
    labels_txt = open(file, 'r')
    
    expected_outputs = []

    lines = []
    for line in labels_txt:
        lines.append(line[:-1])
        
        expected_outputs.append(letra_para_codigo(line[:-1]))
        
    outputs_list = []

    for cod in expected_outputs[:-1]:
        outputs = [0] * len(set(expected_outputs))
        outputs[cod] = 1

        outputs_list.append(outputs)
        
    return outputs_list
    
    
if __name__ == '__main__':
        
    inputs = prepara_features('CARACTERES COMPLETO\\X.txt')
    
    expected_outputs = prepara_labels('CARACTERES COMPLETO\\Y_letra.txt')
        
    layers_size = [len(inputs[0]), 50, len(expected_outputs[0])]

    nn = MLP(layers_size)
    
    for epoch in range(1,20):
        print(f"===================EPOCA {epoch}====================")
        
        for data, label in zip(inputs, expected_outputs):
            #print(f'Input: {data}')
            saidas, erro = nn.train(data, label, learning_rate=0.2)
            if epoch % 5 == 1: 
                print(f'Esperado: {label}')
                print(f'SAIDA: {saidas}')
                print()

    
