import numpy as np
from Activations import Activation
from numpy import ndarray
import pickle
import time

class Perceptron:
    def __init__(self, actvation, n_weights, weights=None):
        self.n_weights = n_weights
        self.count = 0

        if weights is None:
            self.weights = (2 * np.random.rand(n_weights).astype(np.float64)) - 1
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

            case 'linear':
                self.activation = Activation.linear

            case _:
                pass

    def process_input(self, inputs: ndarray) -> float:
        soma = np.dot(self.weights, inputs)

        return self.activation(soma)

class MLP:
    def __init__(self, layers_size):
        self.network = []

        for i in range(1, len(layers_size)):
            self.add_layer(n_inputs=layers_size[i-1]+1, 
                           number_neurons=layers_size[i],
                           activation='relu')

    def foward(self, input_data):
        input_values = input_data.copy()  # Cria uma cópia da lista original
        input_values.append(1)
        input_values = np.array(input_values, dtype=np.float64)

        for layer in self.network:  # itera sobre as camadas excluindo a de entrada
            print(f'input: {input_values}')
            layer_output_values = []

            for neuron in layer:
                layer_output_values.append(neuron.process_input(input_values))
            
            # seta input_values que será usado na proxima camada
            layer_output_values.append(1) # inserindo bias para proxima camada
            print(f'Saída: {layer_output_values}')
            input_values = np.array(layer_output_values, dtype=np.float64)
            print()
        return layer_output_values

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

    def treinar(self, epochs, training_data, validation_data=None, learning_rate=1, parada_cedo=-1):
        if validation_data is None:
            validation_data = training_data
        menor_erro = float('inf')
        melhor_instancia = []  # salva a melhor instancia para caso de parada cedo

        for epoch in range(epochs):
            # treinar
            erro = 2  # mudar erro para receber o erro do treino

            print('Epoch {}/{}\t erro:{}'.format(epoch + 1, epochs, erro))

            if parada_cedo != -1:
                if menor_erro > erro:
                    menor_erro = erro
                    melhor_instancia = pickle.dumps(self.network)  # salva a melhor instancia

                elif parada_cedo == 0:
                    print("Numero maximo de interações sem melhoras atingido!\nInterações:{}"
                          "\t erro:{} ".format(epoch, menor_erro))
                    self.network = pickle.loads(melhor_instancia)  # carrega a melhor instancia

                else:
                    parada_cedo -= 1

if __name__ == '__main__':
    
    values = [1, 2, 3, 4, 5]
    
    layers_size = [len(values), 5,3]

    nn = MLP(layers_size)
    
    start_time = time.time()
    for _ in range(1):
        output = nn.foward(values)
    end_time = time.time()
    
    avarage_time = (end_time-start_time)/1
    
    print()
    print(f"tempo medio {avarage_time} s")
    print(f'Saida da rede: {output}')

    
