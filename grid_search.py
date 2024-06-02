import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import pickle
from Neural_Network import MLP
from data_manipulation import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def grid_search(train_inputs, train_outputs, param_grid, early_stop_epochs=0, k_folds=5):
    best_accuracy = 0
    best_params = None
    results = []

    for params in param_grid:
        layers_size = [len(train_inputs[0])] + params['hidden_layer'] + [len(train_outputs[0])]
        activation = params['activation']
        learning_rate = params['learning_rate']
        max_epochs = params['epochs']
        
        print("Treinando com os parâmetros: ")
        pprint(params)
        
        network = MLP(layers_size, activation)

        # Realiza a validação cruzada k-fold
        mean_accuracy = network.fit_cross_validation(train_inputs, train_outputs, k_folds, learning_rate, max_epochs, early_stop_epochs)

        results.append({
            'hidden_layer': str(params['hidden_layer']),
            'activation': activation,
            'learning_rate': learning_rate,
            'epochs': max_epochs,
            'mean_accuracy': mean_accuracy
        })

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = params

    results_df = pd.DataFrame(results)
    
    print(f"Melhores parâmetros: {best_params}")
    print(f"Melhor acurácia da validação cruzada: {best_accuracy * 100:.2f}%")
    
    # Plotando resultados
    plot_grid_search_results(results_df)
    
    return results_df

def plot_grid_search_results(results_df):
    # Plotar as acurácias para diferentes configurações de parâmetros
    plt.figure(figsize=(12, 8))
    for activation in results_df['activation'].unique():
        subset = results_df[results_df['activation'] == activation]
        plt.plot(subset['epochs'], subset['mean_accuracy'], marker='o', label=activation)
        
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia Média')
    plt.title('Resultados da Busca em Grade')
    plt.legend(title='Função de Ativação')
    plt.grid(True)
    plt.show()
    
    # Plotar os resultados como um gráfico de barras
    results_df.plot(kind='bar', x='hidden_layer', y='mean_accuracy', legend=False)
    plt.xlabel('Configuração')
    plt.ylabel('Acurácia Média')
    plt.title('Acurácias Médias da Busca em Grade')
    plt.xticks(rotation=45)
    plt.show()

def print_grid_search_results(results_df):
    print(results_df)
    print("\nMelhor conjunto de parâmetros:")
    print(results_df.loc[results_df['mean_accuracy'].idxmax()])
