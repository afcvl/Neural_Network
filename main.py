import numpy as np
from Activations import Activation
from numpy import ndarray
import pickle
import time
import matplotlib.pyplot as plt

# Converte uma letra maiúscula do alfabeto (de 'A' a 'Z') para um número correspondente à 
# sua posição no alfabeto, onde 'A' é 1, 'B' é 2, e assim por diante, até 'Z' que é 26. 
# Se a letra fornecida estiver fora do intervalo de 'A' a 'Z', a função imprime uma mensagem de erro e retorna None.
def letra_para_codigo(letra):
    if 'A' <= letra <= 'Z':
        return ord(letra) - 64
    else:
        print("Valor não reconhecido.")
        return None

# Lê cada linha do arquivo, que é dividida em uma lista de elementos usando 
# a vírgula como delimitador. Removemos os espaço em branco a esquerda dos elementos e,
#todos os elementos da linha, exceto o último, são adicionados à lista inputs
def prepara_entradas(file):
    inputs = []
    features_txt = open(file, 'r')
    for line in features_txt:
        line = line.split(',')
        line = [x.lstrip() for x in line]
        inputs.append(line[:-1])
    
    return inputs

# Lê um arquivo contendo rótulos esperados, converte esses rótulos em códigos numéricos usando a função letra_para_codigo, 
# cria uma codificação one-hot-enconding desses rótulos
def prepara_saidas_esperadas(file):
    labels_txt = open(file, 'r')
    
    expected_outputs = []

    lines = []
    for line in labels_txt:
        lines.append(line[0])
        expected_outputs.append(letra_para_codigo(line[0]))
        
    outputs_list = []

    for cod in expected_outputs:
        outputs = [0] * (len(set(expected_outputs))+1)
        outputs[cod] = 1

        outputs_list.append(outputs)
        
    return outputs_list

    #Separa os dados em treinamento e teste com base em uma quantidade especificada de amostras de teste.
def separa_dados(inputs, outputs, num_test_samples):
   
    train_inputs = inputs[:-num_test_samples]
    test_inputs = inputs[-num_test_samples:]
    
    train_outputs = outputs[:-num_test_samples]
    test_outputs = outputs[-num_test_samples:]
    return train_inputs, test_inputs, train_outputs, test_outputs


 #Retorna o erro quadrático médio
def mse(y_true, y_pred):   
    return np.mean((y_true - y_pred)**2)

 # Retorna a derivada do erro quadratico médio
def d_mse(y_true, y_pred):  
    return 2 * (y_pred - y_true)

#Classe que define um perceptron
class Perceptron:
    def __init__(self, actvation, n_weights, weights=None):
        self.n_weights = n_weights + 1   #adição do peso do bias
        self.last_input = None    # para calculo do backpropagation
        self.soma = 0         # para calculo do backpropagation

        if weights is None:
           # Gera valores de pesos entre -0.5 e 0.5
         self.weights = ((np.random.rand(n_weights + 1).astype(np.float64)) - 0.5)
        
        #Caso os pesoss sejam especificados manualmente para fins de teste
        else:
            self.weights = weights
        
        #Define os tipos possíveis de funções de ativação 
        match actvation:
            case 'relu':
                self.activation = Activation.relu
                self.d_activation = Activation.d_relu

            case 'sigmoid':
                self.activation = Activation.sigmoid
                self.d_activation = Activation.d_sigmoid

            case 'tanh':
                self.activation = Activation.tanh
                self.d_activation = Activation.d_tanh

            case 'linear':
                self.activation = Activation.linear

            case _:
                pass

    def process_input(self, inputs: ndarray) -> float:
        
        self.last_input = np.array(inputs, dtype=np.float64)  # salva ultimo input
        self.soma = np.dot(self.weights, inputs)  #  Calcula a soma ponderada e salva resultado da função de transição  
        
        return self.activation(self.soma) #  Aplica a função de ativação na soma ponderada 
    
    
    def compute_gradient(self, d_error):
        # Calcula o gradiente da função de ativação multiplicando a derivada do erro, a derivada da função de ativação e o último vetor de entrada.
        return d_error * self.d_activation(self.soma) * self.last_input

class MLP:
    # O construtor inicializa a rede (self.network) como uma lista vazia e adiciona camadas conforme especificado em layers_size
    def __init__(self, layers_size, activation):
        self.network = []

        for i in range(1, len(layers_size)):
            self.add_layer(n_inputs=layers_size[i-1], 
                           number_neurons=layers_size[i],
                           activation=activation)

     # Cria uma lista de neurônios (Perceptron) e adiciona essa lista à rede.
     # n_inputs: número de entradas para os neurônios nessa camada.
     # number_neurons: número de neurônios nesta camada.
     # activation: função de ativação para os neurônios.
    def add_layer(self, n_inputs, number_neurons, activation):
        neurons_list = []

        for _ in range(number_neurons):
            neurons_list.append(Perceptron(activation, n_inputs))

        self.network.append(neurons_list)
    
    # Propaga os dados de entrada através de cada camada da rede, 
    # aplicando a função de ativação em cada neurônio e passando os resultados para a próxima camada.
    def forward(self, input_data):
        input_values = input_data.copy()  # Cria uma cópia da lista original
        input_values.append(1) # valor 1 para representar o bias 
        input_values = np.array(input_values, dtype=np.float64) #converte a lista de entrada em um array do tipo float64

        for layer in self.network:  # armazenar a saída dos neurônios da camada atual
            layer_output_values = []

            for neuron in layer:
                layer_output_values.append(neuron.process_input(input_values)) # cálculo da saída do neurônio
            
            layer_output_values.append(1) # inserindo bias para proxima camada
            input_values = np.array(layer_output_values, dtype=np.float64) # seta input_values que será usado na proxima camada
  
        return layer_output_values[:-1] # Retira o bias da lista
   
   # Realiza o processo de feed foward e de backpropagation, onde:
   #inputs: dados de entrada para a rede
   #expected_output: a saída esperada ou target para os dados de entrada
   #learning_rate:  taxa de aprendizado usada para ajustar os pesos dos neurônios
    def train_step(self, inputs, expected_output, learning_rate):

        # Feedforward
        # Propaga os dados de entrada através da rede para calcular a saída
        outputs = self.forward(inputs)
        # Converte a saída da rede para um array numpy
        outputs = np.array(outputs)
         
        # Calcula a derivada do erro em relação a saída da da rede usando a função d_mse (derivada do erro quadrático médio).
        error = d_mse(outputs, expected_output)  # calcula derivada do erro para todos os outputs previamente

        # Backpropagation
        # Itera sobre as camadas da rede  em ordem reversa (da última camada para a primeira).
        # Para cada camada, inicializa um vetor next_error para armazenar o erro que será propagado para a próxima camada (anterior)
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            next_error = np.zeros_like(layer[0].weights)

            # Para cada neurônio na camada atual, calcula o gradiente dos pesos e ajusta os pesos com base na taxa de aprendizado e nos gradientes.
            # Prepara o erro para a próxima camada (anterior) somando os produtos dos pesos dos neurônios e os erros derivados.
            for j, neuron in enumerate(layer):
                d_error = error[j]
                gradients = neuron.compute_gradient(d_error)  
                
                # Preparar o erro para a próxima camada
                if i != 0:
                    next_error += neuron.weights * d_error
                    
                # Atualização dos pesos
                neuron.weights += learning_rate * gradients

            error = next_error # propaga o erro para a próxima camada

        # calcula a saída da rede após a alteração dos pesos    
        outputs = self.forward(inputs)
        
        outputs = np.array(outputs)
        
        return outputs.tolist(), mse(expected_output, outputs)

    def train(self, train_inputs, train_outputs, epochs, learning_rate, val_inputs = None,val_outpus = None, early_stop = 0):
        least_erro = float('inf')
        best_epoch = 1 
        best_network =  pickle.dumps(self.network) # salva a rede neural em ram
        error_history = []
        validation = False
        validation_accuracy = 0
        best_accurary = 0
        if val_inputs !=None and val_outpus != None:
            validation = True

        for epoch in range(1, epochs + 1):
            total_error = 0
            acertos = []
            for data, label in zip(train_inputs, train_outputs):
                saidas, erro = self.train_step(data, label, learning_rate)
                total_error += erro
                
                # label = np.array(label)
                # saidas = np.array(saidas)
                # print(f'Esperado: {label.argmax()}')
                # print(f'SAIDA: {saidas.argmax()}')
                # print()

            #total_error+= epoch #usado para forçar a parada antecipada em teste
            #print("!!!!!teste de parada antecipada!!!!!!) #usado para mostrar q está em teste, manter comentado ou descomentado junto com a linha de cima!

            print(f"Época {epoch} - Erro: {total_error}")
            error_history.append(total_error)

            if validation:# Validar a rede neural
                val_predictions = []
                for data in val_inputs:
                    prediction = self.forward(data)
                    val_predictions.append(prediction)

                if len(val_outpus) != len(val_predictions):
                    print(f"Erro: tamanhos incompatíveis. Labels: {len(val_outpus)}, Predictions: {len(val_predictions)}")
                    continue
                
                validation_accuracy = accuracy(val_outpus, val_predictions)

                #validation_accuracy -= 0.01 * epoch  #usado para força a parada cedo no caso de validação
                #print("!!!!!teste de parada antecipada!!!!!!) #usado para mostrar q está em teste, manter comentado ou descomentado junto com a linha de cima!")
                print("Acuracia: ",validation_accuracy)
                

            if early_stop !=0:#parada cedo
                if (validation == True and validation_accuracy < best_accurary) or (validation == False and least_erro < total_error):
                    if best_epoch <= epoch - early_stop:
                        if validation:
                            print("Parada Antecipada!!! A melhor acuracia foi na Época {} e o valor foi {}".format(best_epoch,validation_accuracy))
                        else:
                            print("Parada Antecipada!!! O menor erro foi na Época {} e o valor foi {}".format(best_epoch,least_erro))
                        self.network = pickle.loads(best_network)  # recupera a melhor rede neural da ram
                        break;
                else:
                    least_erro = total_error
                    best_accurary = validation_accuracy
                    best_epoch = epoch
                    best_network = pickle.dumps(self.network) # salva a rede neural em ram

        
        return error_history

# Salva o estado atual da rede neural (os pesos, a estrutura da rede, etc.) em um arquivo
    def save_network(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.network, f)

# Carrega o estado da rede neural a partir de um arquivo previamente salvo
    def load_network(self, file_name):
        with open(file_name, 'rb') as f:
            self.network = pickle.load(f)  

def create_folds(data, labels, k=6):
    num_samples = len(data)
    fold_size = num_samples // k
    remainder = num_samples % k
    
    folds = []
    start_idx = 0
    
    for i in range(k):
        end_idx = start_idx + fold_size + (1 if i < remainder else 0)  # Adiciona 1 aos primeiros 'remainder' folds
        fold_data = data[start_idx:end_idx]
        fold_labels = labels[start_idx:end_idx]
        folds.append((fold_data, fold_labels))
        start_idx = end_idx
    
        print(f"Fold {i + 1}: {len(fold_data)} elementos")
        
    
    return folds

def accuracy(y_true, y_pred):
    correct = np.sum(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    total = len(y_true)
    return correct / total

def validacao_cruzada(data, labels, k=6, epochs=100, learning_rate=0.01):
    # Cria os folds
    folds = create_folds(data, labels, k)
    
    all_val_accuracies = []

    for i in range(k):
        print(f"Treinando e validando com o fold {i + 1}")
        
        # Preparar dados de treinamento e validação
        val_data, val_labels = folds[i]  # Dados de validação do fold atual
        
        train_data = []
        train_labels = []
        
        # Dados de treinamento são todos os folds, exceto o atual
        for j in range(k):
            if j != i:
                train_data.extend(folds[j][0])
                train_labels.extend(folds[j][1])
        
        # Inicializar a rede neural
        mlp = MLP(layers_size0, 'sigmoid')
        
        # Treinar a rede neural
        mlp.train(train_data, train_labels, epochs, learning_rate, early_stop = 5, val_inputs = val_data , val_outpus = val_labels)
        
        # Validar a rede neural
        val_predictions = []
        for data in val_data:
            prediction = mlp.forward(data)
            val_predictions.append(prediction)
        
        # Verifique se o número de rótulos de validação e previsões é igual
        if len(val_labels) != len(val_predictions):
            print(f"Erro: tamanhos incompatíveis. Labels: {len(val_labels)}, Predictions: {len(val_predictions)}")
            continue
        
        fold_accuracy = accuracy(val_labels, val_predictions)
        all_val_accuracies.append(fold_accuracy)
        
        print(f"Acurácia de validação no fold {i + 1}: {fold_accuracy * 100:.2f}%")
    
    # Calcula a média da acurácia de validação sobre todos os folds
    mean_cross_val_accuracy = np.mean(all_val_accuracies)
    print(f"Acurácia média da validação cruzada média: {mean_cross_val_accuracy * 100:.2f}%")


    return mean_cross_val_accuracy



if __name__ == '__main__':

  #Com cross validation
    inputs_train = prepara_entradas('CARACTERES COMPLETO/X.txt')
    outputs_train = prepara_saidas_esperadas('CARACTERES COMPLETO/Y_letra.txt')
    
    #layers_size0 = [len(inputs_train[0]), 50, len(outputs_train[0])]
    
    #mean_accuracy = validacao_cruzada(inputs_train, outputs_train, k=4, epochs=30, learning_rate=0.05)
   
    n_neurons = [20,30,40,50,60,70,80]
    result = []
    
    for i in n_neurons:    
        layers_size0 = [len(inputs_train[0]), i, len(outputs_train[0])]
        mean_accuracy = validacao_cruzada(inputs_train, outputs_train, k=5, epochs=50, learning_rate=0.05)
        result.append(mean_accuracy)
    
    print()
    for i, n in enumerate(n_neurons):
        print(f'{n} neuronios -> Acuracia {result[i]}')
   # Sem cross validation
    
    # # Separar dados de treinamento e teste
    # num_test_samples = 130
    # train_inputs, test_inputs, train_outputs, test_outputs = separa_dados(inputs_train, outputs_train, num_test_samples)
    
    # layers_size = [len(train_inputs[0]), 40, len(train_outputs[0])]
    # nn = MLP(layers_size, activation='sigmoid')
    
    # # Treinar a rede
    # epochs = 40
    # learning_rate = 0.005
    # error_history = nn.train(train_inputs, train_outputs, epochs, learning_rate)
    
    # # Salvar a rede treinada
    # nn.save_network('network.nw')
    
    # # Criar gráfico do histórico de erro
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(error_history, label='train_error')
    # ax.legend()
    # plt.show()
    
    
        


