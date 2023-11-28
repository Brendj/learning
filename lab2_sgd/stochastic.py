import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# Функция активации сигмоида.
def sigmoid_activation(x):
    return 1 / (1 + math.exp(-x))

# Производная функции активации.
def sigmoid_derivative_activation(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

# Возвденние в квадрат
def square(x):
    return x * x

square = np.vectorize(square)

size_input = 4
first_layer = 6
second_layer = 3
gradient_step = 0.08

def errors(result, expected_class):
    expected_vector = np.array([0, 0, 0])
    expected_vector[expected_class] = 1
    # считаем ошибку по MSE для каждого класса и складываем
    sum_errors = sum(square(result - expected_vector))
    return (sum_errors, np.array([result - expected_vector]))

df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:150, 4].values
y = np.where(y == "Iris-setosa", 0, y)
y = np.where(y == "Iris-versicolor", 1, y)
y = np.where(y == "Iris-virginica", 2, y)

X = df.iloc[0:150, 0:4].values

# матрица весов первого слоя
W_1 = np.random.rand(size_input, first_layer)
# матрица весов вторго  слоя
W_2 = np.random.randn(first_layer, second_layer)

vector_sigmoid_activation = np.vectorize(sigmoid_activation)
vector_sigmoid_derivative_activation = np.vectorize(sigmoid_derivative_activation)

ziped = zip(X, y)
zipped_list = list(ziped)
epoch = 100

errors_over_time = []

for i in range(epoch):
    sum_errors = 0
    for x_input, expected in zipped_list:
        o1 = x_input @ W_1
        o1_activated = vector_sigmoid_activation(o1)
        o2 = o1_activated @ W_2
        o2_activated = vector_sigmoid_activation(o2)
        error, expected_vector = errors(o2_activated, expected)
        sum_errors = sum_errors + error

        de_e = vector_sigmoid_derivative_activation(o2) * expected_vector
        de_dw2 = np.dot(np.array([o1_activated]).T, de_e)
        de_o1_activated = de_e @ W_2.T
        de_01 = vector_sigmoid_derivative_activation(o1) * de_o1_activated
        de_dw1 = np.array([x_input]).T @ de_01
        W_1 = W_1 - gradient_step * de_dw1
        W_2 = W_2 - gradient_step * de_dw2

    errors_over_time.append(sum_errors / len(zipped_list))
    print('sumary errors', sum_errors)
    print('sumary errors mean', sum_errors / len(zipped_list))

plt.plot(errors_over_time)
plt.title('Ошибка обучения')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка')
plt.show()

correct = 0
incorrect = 0
for x_input, expected in zipped_list:
    o1 = x_input @ W_1
    o1_activated = vector_sigmoid_activation(o1)
    o2 = o1_activated @ W_2
    o2_activated = vector_sigmoid_activation(o2)
    error, expected_vector = errors(o2_activated, expected)
    sum_errors = sum_errors + error
    if (np.argmax(o2_activated, axis=0) == expected):
        correct += 1
    else:
        incorrect += 1

print('correct', correct)
print('incorrect', incorrect)
