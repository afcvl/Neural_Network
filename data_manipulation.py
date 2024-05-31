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
