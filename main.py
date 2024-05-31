import numpy as np
from pprint import pprint
import time
import matplotlib.pyplot as plt
from data_manipulation import *
from Neural_Network import MLP

def grid_search(train_inputs, train_outputs, param_grid, early_stop_epochs = 0, k_folds=5):
    best_accuracy = 0
    best_params = None
    results = []

    for params in param_grid:
        layers_size = [len(train_inputs[0])] + params['hidden_layer'] + [len(train_outputs[0])]
        activation = params['activation']
        learning_rate = params['learning_rate']
        max_epochs = params['epochs']
        
        print("Training with parameters: ")
        pprint(params)
        
        network = MLP(layers_size, activation)

        # Perform k-fold cross-validation
        mean_accuracy = network.fit_cross_validation(train_inputs, train_outputs, k_folds, learning_rate, max_epochs,early_stop_epochs)

        results.append({
            'params': params,
            'mean_accuracy': mean_accuracy
        })

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_accuracy * 100:.2f}%")

    return results


if __name__ == '__main__':

    #  Cenário 0: MLP crua, sem validação e otimização dos parâmetros (valores aleatórios)

    #  Descomentar linhas 195-199 do arquivo "Neural_Nerwork.py" para visualizar a rede treinando

    # DADOS DE TREINAMENTO: 1196 PRIMEIROS

    # DADOS DE TESTE: 130 ÚLTIMOS
    
    inputs_train = prepara_entradas('CARACTERES COMPLETO/X.txt')
    outputs_train = prepara_saidas_esperadas('CARACTERES COMPLETO/Y_letra.txt')
    
    # # Separar inicialmente dados de treinamento e teste
    num_test_samples = 130
    inputs_train, test_inputs, outputs_train, test_outputs = separa_dados(inputs_train, outputs_train, num_test_samples)

    param_grid = [
    {'hidden_layer': [50], 'activation': 'sigmoid', 'learning_rate': 0.05, 'epochs': 50},
    {'hidden_layer': [150], 'activation': 'sigmoid', 'learning_rate': 0.005, 'epochs': 50},
    # Add more combinations as needed
    ]

    # Faz o grid_search para o treinamento da rede
    results = grid_search(inputs_train, outputs_train, param_grid, early_stop_epochs=5, k_folds=5)
    
    print()
    print('------------------- Resultado GridSeach ---------------------')
    pprint(results)
    print()
    
   
   
    # =================== Treina rede com cross validation =============== 
    print('------------------ CROSS VALIDATION SIMPLES -------------')
    layers_size = [len(inputs_train[0]), 50, len(outputs_train[0])]
        
    mlp = MLP(layers_size, 'sigmoid')
        
    mean_accuracy = mlp.fit_cross_validation(data=inputs_train,
                                                labels=outputs_train,
                                                n_folds=5,
                                                lr=0.03,
                                                max_epochs=40,
                                                early_stop_epochs=5)
        
    # testa a rede nos dados de teste separados inicialmente
    cont = 0
    for data, label in zip(test_inputs, test_outputs):
        pred = mlp.forward(data)
        pred = np.array(pred)
        label = np.array(label)
        if pred.argmax() ==  label.argmax():
            cont += 1
    
    print()
    print(f'Acuracia conjuto de testes: {cont/len(test_inputs)}')