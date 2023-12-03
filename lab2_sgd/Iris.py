import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
df = pd.read_csv('data.csv')

# Кодирование меток классов
le = LabelEncoder()
y = le.fit_transform(df.iloc[:, 4].values).reshape(-1,1)

# Выбор всех четырех признаков
X = df.iloc[:, [0, 1, 2, 3]].values

# Добавление единиц к X
X = np.concatenate([np.ones((len(X),1)), X], axis = 1)

# Определение размеров входного, скрытого и выходного слоя
inputSize = X.shape[1]
hiddenSizes = 5
outputSize = len(np.unique(y))  # количество уникальных классов

# Инициализация весов
weights = [
    np.random.uniform(-2, 2, size=(inputSize,hiddenSizes)),
    np.random.uniform(-2, 2, size=(hiddenSizes,outputSize))
]

# Определение функций активации и их производных
def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def derivative_sigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))

# Определение функций прямого и обратного распространения
def feed_forward(x):
    input_ = x
    hidden_ = sigmoid(np.dot(input_, weights[0]))
    output_ = sigmoid(np.dot(hidden_, weights[1]))
    return [input_, hidden_, output_]

def backward(learning_rate, target, net_output, layers):
    err = (target - net_output)
    for i in range(len(layers)-1, 0, -1):
        err_delta = err * derivative_sigmoid(layers[i])
        err = np.dot(err_delta, weights[i - 1].T)
        dw = np.dot(layers[i - 1].T, err_delta)
        weights[i - 1] += learning_rate * dw

# Определение функций обучения и предсказания
def train(x_values, target, learning_rate):
    output = feed_forward(x_values)
    backward(learning_rate, target, output[2], output)

def predict(x_values):
    return feed_forward(x_values)[-1]

# Обучение нейронной сети
iterations = 50
learning_rate = 0.01

for i in range(iterations):
    train(X, y, learning_rate)
    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - predict(X)))))

# Предсказание
pr = predict(X)
print(sum(abs(y-(pr>0.5))))
