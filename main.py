import numpy as np
from pprint import pprint
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from data_manipulation import *
from Neural_Network import MLP
from grid_search import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

if __name__ == '__main__':

    #  Cenário 0: MLP crua, sem validação e otimização dos parâmetros (valores aleatórios)

    #  Descomentar linhas 195-199 do arquivo "Neural_Nerwork.py" para visualizar a rede treinando

    # DADOS DE TREINAMENTO: 1196 PRIMEIROS

    # DADOS DE TESTE: 130 ÚLTIMOS
    
    inputs_train_complete = prepara_entradas('CARACTERES COMPLETO/X.txt')
    outputs_train_complete = prepara_saidas_esperadas('CARACTERES COMPLETO/Y_letra.txt')

    inputs_train = prepara_entradas('CARACTERES COMPLETO/X.txt')
    outputs_train = prepara_saidas_esperadas('CARACTERES COMPLETO/Y_letra.txt')
    
    # # Separar inicialmente dados de treinamento e teste
    num_test_samples = 130
    inputs_train, test_inputs, outputs_train, test_outputs = separa_dados(inputs_train, outputs_train, num_test_samples)

    param_grid = [
    {'hidden_layer': [200], 'activation': 'sigmoid', 'learning_rate': 0.05, 'epochs': 100},
    # {'hidden_layer': [200], 'activation': 'sigmoid', 'learning_rate': 0.01, 'epochs': 50},
    # {'hidden_layer': [200], 'activation': 'sigmoid', 'learning_rate': 0.01, 'epochs': 80},
    # {'hidden_layer': [200], 'activation': 'sigmoid', 'learning_rate': 0.05, 'epochs': 80},
    # {'hidden_layer': [250], 'activation': 'sigmoid', 'learning_rate': 0.05, 'epochs': 80},
    ]


    # Realiza o grid_search para o treinamento da rede
    results = grid_search(inputs_train, outputs_train, param_grid, early_stop_epochs=5, k_folds=5)
    
    print()
    print('------------------- Resultado Grid Search ---------------------')
    print_grid_search_results(results)
    
   
    # =================== Treina rede com cross validation =============== 
    print('------------------ CROSS VALIDATION SIMPLES -------------')
    layers_size = [len(inputs_train[0]), 50, len(outputs_train[0])]
        
    mlp_cv = MLP(layers_size, 'sigmoid')
        
    mean_accuracy = mlp_cv.fit_cross_validation(data=inputs_train,
                                                labels=outputs_train,
                                                n_folds=5,
                                                lr=0.03,
                                                max_epochs=40,
                                                early_stop_epochs=5)


        
    # testa a rede nos dados de teste separados inicialmente e calcula a matriz de confusão
    y_true_cv = []
    y_pred_cv = []
    cont = 0
    for data, label in zip(test_inputs, test_outputs):
        pred = mlp_cv.forward(data)
        pred = np.array(pred)
        label = np.array(label)
        y_true_cv.append(label.argmax())
        y_pred_cv.append(pred.argmax())
        if pred.argmax() ==  label.argmax():
            cont += 1
    
    plot_confusion_matrix(y_true_cv, y_pred_cv, "Matriz de Confusão -  Modelo com Cross Validation")

    # print()
    # print(f'Acuracia conjuto de testes: {cont/len(test_inputs)}')

    # =================== Treina rede com holdout e random sampling =============== 

    # layers_size = [len(inputs_train_complete[0]), 50, len(outputs_train_complete[0])]

    # print(f'Acuracia conjuto de testes: {cont/len(test_inputs)}')