import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')  # Загрузка данных из файла CSV
y = df.iloc[:, 4].values  # Выбор меток классов
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)  # Преобразование меток классов в числовой формат
X = df.iloc[:, [0, 2]].values  # Выбор признаков
X = np.concatenate([np.ones((len(X),1)), X], axis = 1)  # Добавление единиц к признакам для учета смещения

def sigmoid(y):  # Определение функции активации сигмоиды
    return 1 / (1 + np.exp(-y))

def derivative_sigmoid(y):  # Определение производной функции активации сигмоиды
    return sigmoid(y) * (1 - sigmoid(y))

inputSize = X.shape[1]  # Определение размера входного слоя
hiddenSizes = 5  # Определение размера скрытого слоя
outputSize = 1 if len(y.shape) else y.shape[1]  # Определение размера выходного слоя

weights = [  # Инициализация весов
    np.random.uniform(-2, 2, size=(inputSize,hiddenSizes)),
    np.random.uniform(-2, 2, size=(hiddenSizes,outputSize))
]

def feed_forward(x):  # Определение функции прямого распространения
    input_ = x
    hidden_ = sigmoid(np.dot(input_, weights[0]))
    output_ = sigmoid(np.dot(hidden_, weights[1]))
    return [input_, hidden_, output_]

def train_SGD(x_values, target, learning_rate):  # Определение функции обучения с использованием стохастического градиентного спуска
    for i in range(len(x_values)):
        output = feed_forward([x_values[i]])
        backward(learning_rate, target[i], output[2], output)
    return None

def backward(learning_rate, target, net_output, layers):  # Определение функции обратного распространения ошибки
    err = (target - net_output)
    for i in range(len(layers)-1, 0, -1):
        err_delta = err * derivative_sigmoid(layers[i])
        err = np.dot(err_delta, weights[i - 1].T)
        dw = np.dot(np.array(layers[i - 1]).reshape(-1, 1), err_delta)
        weights[i - 1] += learning_rate * dw

def predict(x_values):  # Определение функции предсказания
    return feed_forward(x_values)[-1]

iterations = 50  # Количество итераций обучения
learning_rate = 0.01  # Скорость обучения

errors = []  # Создание списка для хранения ошибок на каждой итерации

for i in range(iterations):  # Цикл обучения
    train_SGD(X, y, learning_rate)
    error = np.mean(np.square(y - predict(X)))  # Вычисление среднеквадратичной ошибки
    errors.append(error)  # Добавление ошибки в список
    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(error))

# Построение графика уменьшения ошибки
plt.plot(errors)
plt.xlabel('Итерация')
plt.ylabel('Средняя ошибка')
plt.title('Уменьшение ошибки в процессе обучения')
plt.show()
pr = predict(X)  # Предсказание на обучающем наборе
print(sum(abs(y-(pr>0.5))))  # Вывод ошибки классификации
