from Activations import Activation
import pickle
import numpy as np
from numpy import ndarray

 #Retorna o erro quadrático médio
def mse(y_true, y_pred):   
    if isinstance(y_true, list):
        y_true = np.array(y_true)
        
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
        
    return np.mean((y_true - y_pred)**2)

# Retorna a derivada do erro quadratico médio
def d_mse(y_true, y_pred):  
    return 2 * (y_pred - y_true)

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

#Classe que define o Neurônio
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

            case _:
                pass

    def process_input(self, inputs: ndarray) -> float:
        
        self.last_input = np.array(inputs, dtype=np.float64)  # Salva ultimo input
        self.soma = np.dot(self.weights, inputs)  #  Calcula a soma ponderada e salva resultado da função de transição  
        
        return self.activation(self.soma) #  Aplica a função de ativação na soma ponderada 
    
    
    def compute_gradient(self, d_error):
        # Calcula o gradiente da função de ativação multiplicando a derivada do erro, a derivada da função de ativação e o último vetor de entrada.
        return d_error * self.d_activation(self.soma) * self.last_input

    #reinicia os pesos aletoriamente
    def reset_weights(self):
        self.weights = ((np.random.rand(len(self.weights)).astype(np.float64)) - 0.5)
        
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
        
    def reset_net_weights(self):
        for layer in self.network:
            for neuron in layer:
                neuron.reset_weights()
    
    # Propaga os dados de entrada através de cada camada da rede, 
    # aplicando a função de ativação em cada neurônio e passando os resultados para a próxima camada.
    def forward(self, input_data):
        input_values = input_data.copy()  # Cria uma cópia da lista original
        input_values.append(1) # valor 1 para o bias 
        input_values = np.array(input_values, dtype=np.float64) #converte a lista de entrada em um array do tipo float64

        for layer in self.network:  
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
        outputs = np.array(outputs)
         
        # calcula derivada do erro para todos os outputs previamente 
        error = d_mse(outputs, expected_output)  

        # Backpropagation
        # Itera sobre as camadas da rede em ordem reversa (da última camada para a primeira).
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

        # # calcula a saída da rede após a alteração dos pesos (comentar para aumentar eficiência da execução)
        # outputs = self.forward(inputs)
        # outputs = np.array(outputs)
        
        #retorna a saída da rede e o erro
        return outputs.tolist(), mse(expected_output, outputs)
    
    def train(self, train_inputs, train_outputs, val_data, val_labels, epochs, learning_rate, early_stop_epochs = 0, early_learning_stop = 0):
        validation_accuracy = 0
        train_error_history = []
        val_error_history = []
        val_accuracy_history = []

        best_epoch = 0
        best_accuracy = 0

        for epoch in range(1, epochs + 1):
            total_train_error = 0

            # percorre os dados, treina a rede para cada um e retorna o erro do treinamento
            for data, label in zip(train_inputs, train_outputs):
                saidas, train_error = self.train_step(data, label, learning_rate)
                total_train_error += train_error
                
                # # -------- DESCOMENTAR PARA VER EXECUÇÃO DA REDE -----------
                # label = np.array(label)
                # saidas = np.array(saidas)
                # print(f'Esperado: {label.argmax()}')
                # print(f'SAIDA: {saidas.argmax()}')
                # print()
                # # -----------------------------------------------------------
                

            total_train_error = total_train_error/len(train_inputs) # descobre erro total médio
            
            train_error_history.append(total_train_error)  # histórico do erro na época

            # Faz as predições em cima dos dados de validação
            val_predictions = []
            for data in val_data:
                prediction = self.forward(data)
                val_predictions.append(prediction)
                
            if len(val_labels) != len(val_predictions):
                print(f"Erro: tamanhos incompatíveis. Labels: {len(val_labels)}, Predictions: {len(val_predictions)}")
                continue
            
            # calcula o erro no conjunto de validação
            val_error =  mse(val_predictions, val_labels)
            
            val_error_history.append(val_error)
            
            # calcula acurácia no conjunto de validação
            validation_accuracy = accuracy(val_labels, val_predictions)
            
            val_accuracy_history.append(validation_accuracy)

            print(f"Epoca {epoch} || Erro treino: {total_train_error:.5f} || Erro validacao: {val_error:.5f} || Acuracia: {validation_accuracy:.4f}")

            # PARADA ANTECIPADA
            if early_stop_epochs > 0:

                if validation_accuracy >= best_accuracy:
                    best_epoch = epoch
                    best_accuracy = validation_accuracy
                    self._save_best()

                elif epoch >= best_epoch + early_stop_epochs:
                    self._load_best()
                    print("Parada Antecipada!!!! Melhor Epoca foi a {}".format(best_epoch,best_accuracy))
                    break

        self._clear__copy()
        return train_error_history, val_error_history
    
    def fit_cross_validation(self, data, labels, n_folds, lr, max_epochs=200, early_stop_epochs = 0):
        
        folds = create_folds(data, labels, n_folds)
        
        all_val_accuracies = []
        
        for valid_fold in range(n_folds):
            self.reset_net_weights()
            
            print(f"\nTreinando e validando com o fold {valid_fold + 1}")
            
            # Preparar dados de treinamento e validação
            val_data, val_labels = folds[valid_fold]  # Dados de validação do fold atual
            
            train_data = []
            train_labels = []
            
            # Dados de treinamento são todos os folds, exceto o atual
            for j in range(len(folds)):
                if j != valid_fold:
                    train_data.extend(folds[j][0])
                    train_labels.extend(folds[j][1])

            # Treinar a rede neural
            # Faz o treinamento passando pelas épocas e retorna o histórico do erro do conjutno de treino e do
            # conjunto de validação
            train_error_history, val_error_history = self.train(train_data, train_labels, val_data, val_labels, max_epochs, lr, early_stop_epochs)
        
            # Aplica a rede treinada no fold de validação
            val_predictions = []
            for data in val_data:
                prediction = self.forward(data)
                val_predictions.append(prediction)
            
            # calcula a acurácia no fold de validação
            fold_accuracy = accuracy(val_labels, val_predictions)
            all_val_accuracies.append(fold_accuracy)
            
            # Verifique se o número de rótulos de validação e previsões é igual
            if len(val_labels) != len(val_predictions):
                print(f"Erro: tamanhos incompatíveis. Labels: {len(val_labels)}, Predictions: {len(val_predictions)}")
                return
            
            print(f"Acurácia de validação no fold {valid_fold + 1}: {fold_accuracy * 100:.2f}%")
    
        # Calcula a média da acurácia de validação sobre todos os folds
        mean_cross_val_accuracy = np.mean(all_val_accuracies)
        print(f"Acurácia média da validação cruzada média: {mean_cross_val_accuracy * 100:.2f}%")

        return mean_cross_val_accuracy
    
# Método .fit_holdout()
    
    def fit_holdout(self, data, labels, test_size=0.33, epochs=50, learning_rate=0.01, early_stop_epochs=0):
        
        # Separa os dados de treinamento e de teste (de forma aleatória), sendo 2/3 dados de treinamento e 1/3 dados de teste
        num_samples = len(data)
        indices = np.random.permutation(num_samples)
        test_set_size = int(num_samples * test_size)

        # Divide os índices dos dados em conjuntos de índices de teste e índices de treinamento
        # Seleciona os primeiros 'test_set_size' índices para o conjunto de teste
        test_indices = indices[:test_set_size]

        # Seleciona os índices restantes para o conjunto de treinamento
        train_indices = indices[test_set_size:]

        # Divide os dados e rótulos em conjuntos de treinamento e teste com base nos índices gerados aleatoriamente
        # Seleciona os dados de treinamento e rótulos de treinamento com base nos índices de treinamento
        train_data = [data[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]

        # Treina a rede neural
        self.train(train_data, train_labels, test_data, test_labels, epochs, learning_rate, early_stop_epochs)

        # Realiza predições para cada amostra nos dados de teste
        test_predictions = [self.forward(data) for data in test_data]
        
        # Calcula a acurácia 
        test_accuracy = accuracy(test_labels, test_predictions)
        print(f"Acurácia no conjunto de teste: {test_accuracy * 100:.2f}%")

        return test_accuracy

    def fit_random_sampling(self, data, labels, test_size=0.33, epochs=50, learning_rate=0.01, k=10, early_stop_epochs=0):
        accuracies = []
        
        # Para cada iteração, realiza um holdout. 
        # É importante realizar o holdout diversas vezes para testar a validação com diferentes conjuntos de dados gerados aleatoriamente
        for i in range(k):
            print(f"Repetição {i+1}/{k}")
            accuracy = self.fit_holdout(data=data, labels=labels, test_size=test_size, epochs=epochs, learning_rate=learning_rate, early_stop_epochs=early_stop_epochs)
            accuracies.append(accuracy)

        # Retorna a média de todos as rodadas de holdout realizadas
        mean_accuracy = np.mean(accuracies)
        print(f"Acurácia média após {k} repetições: {mean_accuracy * 100:.2f}%")

        return mean_accuracy

# Salva o estado atual da rede neural (os pesos, a estrutura da rede, etc.) em um arquivo
    def save_network(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.network, f)

# Carrega o estado da rede neural a partir de um arquivo previamente salvo
    def load_network(self, file_name):
        with open(file_name, 'rb') as f:
            self.network = pickle.load(f)  
    
    def _save_best(self):
        self._copy = pickle.dumps(self.network)

    def _load_best(self):
        self.network = pickle.loads(self._copy)

    def _clear__copy(self):
        del self._copy
