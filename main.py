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
        #  print(self.weights, inputs)
        soma = np.dot(self.weights, inputs)
        return self.activation(soma)


class MLP:
    def __init__(self, layers, activation='relu'):
        """
        self.inputs = []
        self.exits = []
        self.hidden = []

        for i in range(layers[0]):
            self.inputs.append(Perceptron(activation, 0))

        for i in range(layers[1]):
            self.hidden.append(Perceptron(activation, layers[0]))

        for i in range(layers[2]):
            self.exits.append(Perceptron(activation, layers[1]))

        self.network = [self.inputs, self.hidden, self.exits]
        """

        self.network = []
        anterior = 0
        for tamanho in layers:
            #  print(tamanho)
            layers = []
            for i in range(tamanho):
                layers.append(Perceptron(activation, anterior))

            self.network.append(layers)
            anterior = tamanho

    def foward(self, input_data):
        """
        input_data.append(1)  # bias

        output_layer1 = np.array(input_data, dtype=np.float64)
        output_layer2 = []
        output_layer3 = []

        start = time.time()
        for neuron in self.hidden:
            output_layer2.append(neuron.process_input(output_layer1))
        output_layer2.append(1)  # append bias

        output_layer2 = np.array(output_layer2, dtype=np.float64)

        for neuron in self.exits:
            output_layer3.append(neuron.process_input(output_layer2))

        end = time.time()
        duracao = end - start
        print("tempo:",duracao,"s")

        return output_layer3

        """
        input_data.append(1)  # bias
        input_layer = np.array(input_data, dtype=np.float64)
        output_layer = []

        start = time.time()
        first_layer_skipped = False

        for index, layer in enumerate(self.network):
            if not first_layer_skipped:
                first_layer_skipped = True
                continue

            for neuron in layer:
                output_layer.append(neuron.process_input(input_layer))
            output_layer.append(1)  # append bias

            # input_layer para proxima interação ou saida da rede neural
            input_layer = np.array(output_layer, dtype=np.float64)
            output_layer = []

        output_layer = np.delete(input_layer, -1)
        end = time.time()
        duracao = end - start
        print("tempo:", duracao, "s")
        
        return output_layer


if __name__ == '__main__':
    values = [5, 3, 4, 2, 8, 6, 5, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4]
    nn = MLP([len(values), 20000, 50, 20, 1000, 10])
    print("saida:", nn.foward(values))
