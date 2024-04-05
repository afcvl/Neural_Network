import numpy as np
from Activations import Activation
from numpy import ndarray
import pickle


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

            case 'linear':
                self.activation = Activation.linear

            case _:
                pass

    def process_input(self, inputs: ndarray) -> float:
        soma = np.dot(self.weights, inputs)

        return self.activation(soma)


class MLP:
    def __init__(self, file_name='key.bitcoin'):
        self.file_name = file_name
        self.network = []
        self.n_weights = 0

    def foward(self, input_data):
        input_values = input_data.copy()  # Cria uma cópia da lista original
        input_values.append(1)
        input_values = np.array(input_values, dtype=np.float64)
        output_values = []

        for layer in self.network[1:]:  # itera sobre as camadas excluindo a de entrada

            for neuron in layer:
                output_values.append(neuron.process_input(input_values))
            output_values.append(1)  # append bias

            # seta input_values que será usado na proxima camada
            input_values = np.array(output_values, dtype=np.float64)
            output_values = []

        output_values = np.delete(input_values, -1)

        return output_values

    def add(self, number_neurons, activation):
        neurons_list = []

        for _ in range(number_neurons):
            neurons_list.append(Perceptron(activation, self.n_weights))

        self.network.append(neurons_list)
        self.n_weights = number_neurons

    def remove_last(self):
        self.network.pop()

    def salvar(self):
        with open(self.file_name, 'wb') as f:
            pickle.dump(self.network, f)

    def carregar(self):
        with open(self.file_name, 'rb') as f:
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
    layers = [len(values), 5000, 2000, 100, 10]
    nn = MLP()
    for layer in layers:
        nn.add(layer, 'relu')
    nn.add(50, 'linear')
    nn.remove_last()
    nn.salvar()
    saida1 = nn.foward(values)
    # print("saida1:", saida1)
    nn = MLP()
    nn.carregar()
    saida2 = nn.foward(values)
    # print("saida2:", saida2)
    print("diferença:", saida2-saida1)
