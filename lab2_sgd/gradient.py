import pandas as pd
import numpy as np

# Функция активации сигмоида.
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

# Производная функции активации.
def sigmoid_derivative_activation(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

# Ошибка MSE
def errors(result, expected_class):
    expected_vector = np.zeros(3)
    expected_vector[expected_class] = 1
    return np.sum(np.square(result - expected_vector)), result - expected_vector

# Загрузка и подготовка данных
df = pd.read_csv('data.csv')
df = df.sample(frac=1) # перемешивание
y = df.iloc[0:150, 4].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}).values
X = df.iloc[0:150, 0:4].values

# Параметры сети
size_input = 4
first_layer = 6
second_layer = 3
gradient_step = 0.08
epoch = 100

# Инициализация весов
W_1 = np.random.rand(size_input, first_layer)
W_2 = np.random.randn(first_layer, second_layer)

# Обучение
for i in range(epoch):
    sum_errors = 0
    for x_input, expected in zip(X, y):
        # Прямой проход
        o1 = x_input @ W_1
        o1_activated = sigmoid_activation(o1)
        o2 = o1_activated @ W_2
        o2_activated = sigmoid_activation(o2)

        # Вычисление ошибки
        error, expected_vector = errors(o2_activated, expected)
        sum_errors += error

        # Обратное распространение ошибки
        de_e = sigmoid_derivative_activation(o2) * expected_vector
        de_dw2 = np.outer(o1_activated, de_e)
        de_o1_activated = de_e @ W_2.T
        de_01 = sigmoid_derivative_activation(o1) * de_o1_activated
        de_dw1 = np.outer(x_input, de_01)

        # Обновление весов
        W_1 -= gradient_step * de_dw1
        W_2 -= gradient_step * de_dw2

    print(f'Эпоха {i+1}, суммарная ошибка: {sum_errors / len(X)}')

# Тестирование
correct = sum(np.argmax(sigmoid_activation(sigmoid_activation(x_input @ W_1) @ W_2)) == expected for x_input, expected in zip(X, y))
print(f'Точность: {correct / len(X)}')
