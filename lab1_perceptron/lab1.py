# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import os

# Проверка наличия файла
if not os.path.isfile('data.csv'):
    print("Файл 'data.csv' не найден.")
    exit()

# Загрузка данных
df = pd.read_csv('data.csv')

# Проверка структуры данных
if len(df.columns) < 5 or df.shape[0] < 100:
    print("Структура данных не соответствует ожидаемой.")
    exit()

# Перемешивание данных
df = df.iloc[np.random.permutation(len(df))]

# Извлечение меток классов
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)  # Преобразование меток классов

# Извлечение входных данных
X = df.iloc[0:100, [0, 2]].values

# Определение размеров слоев
input_size = X.shape[1]
neurons_hidden_layer= 10
neurons_output_layer = 1 if len(y.shape) else y.shape[1]  # количество выходных сигналов равно количеству классов задачи

# Инициализация весов и порогов первого слоя
W_1 = np.zeros((1 + input_size, neurons_hidden_layer))
W_1[0, :] = (np.random.randint(0, 3, size = (neurons_hidden_layer)))  # пороговые значения
W_1[1:, :] = (np.random.randint(-1, 2, size = (input_size, neurons_hidden_layer)))  # веса

# Инициализация весов и порогов второго слоя
W_2 = np.random.randint(0, 2, size = (1 + neurons_hidden_layer, neurons_output_layer)).astype(np.float64)

# Функция предсказания
def predict(X):
    W_1_out = np.where((np.dot(X, W_1[1:, :]) + W_1[0, :]) >= 0.0, 1, -1).astype(np.float64)
    W_2_out = np.where((np.dot(W_1_out, W_2[1:, :]) + W_2[0, :]) >= 0.0, 1, -1).astype(np.float64)
    return W_2_out, W_1_out

# Параметры обучения
n_iter=0
step = 0.01
check_iter = 5

# Список для хранения  матрицы весов второго слоя
list_w_2_weights= []

# Обучение
while(True):
    n_iter+=1

    # Обновление весов для каждого образца в обучающем наборе
    for x_input, expected, j in zip(X, y, range(X.shape[0])):
        W_2_out, W_1_out = predict(x_input)
        W_2[1:] += ((step * (expected - W_2_out)) * W_1_out).reshape(-1, 1)  # Обновление весов
        W_2[0] += step * (expected - W_2_out)  # Обновление порогового значения

    # Сохранение текущих весов для проверки зацикливания
    list_w_2_weights.append(W_2.tobytes())

    # Проверка наличия ошибок
    W_2_out, W_1_out = predict(X)
    sum_errors =sum(W_2_out.reshape(-1) - y)
    if (sum_errors == 0):
        print('Все примеры обучающей выборки решены:')
        break

    # Проверка зацикливания каждые check_iter итераций
    if ((n_iter % check_iter )==0):
        break_out_flag = False
        for item in list_w_2_weights:
            if list_w_2_weights.count(item) > 1:
                print('Повторение весов:')
                break_out_flag = True
                break
        if break_out_flag:
            break

# Подсчет ошибок после завершения обучения
W_2_out, W_1_out = predict(X)
sum_errors =sum( W_2_out.reshape(-1) - y)
print('sum_errors', sum_errors)
