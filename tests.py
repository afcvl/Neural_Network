import numpy as np
import time

lista_aleatoria = np.random.rand(100000)

arr1 = np.array(lista_aleatoria)
arr2 = np.array(lista_aleatoria)
arr3 = np.array(lista_aleatoria)

start =  time.time()

for i in range(arr1.shape[0]):
    arr3[i] = arr1[i]*arr2[i]
print(arr3)

fim = time.time()

tempo_decorrido = fim - start
print("Tempo decorrido for:", tempo_decorrido, "segundos")



start =  time.time()

print(np.multiply(arr1, arr2))

fim = time.time()

tempo_decorrido = fim - start
print("Tempo decorrido dot:", tempo_decorrido, "segundos")
