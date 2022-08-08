import numpy as np
import tensorflow.keras.datasets.mnist as mnist
import neuralnet as nn


def one_hot(Y):
    num_labels = len(set(Y))
    new_Y = []
    for label in Y:
        encoding = np.zeros(num_labels)
        encoding[label] = 1.
        new_Y.append(encoding)
    return np.array(new_Y)

#Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten input
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Normalize data
x_train = np.array(x_train/255.)
x_test = np.array(x_test/255.)

#Convert target in one-hot encoding
y_train = one_hot(y_train)
y_test = one_hot(y_test)


model = nn.NeuralNetwork()
model.add(nn.Layer(64, activation='leakyrelu'))
model.add(nn.Layer(32, activation='tanh'))
model.add(nn.Layer(10, activation='softmax'))
stop = nn.EarlyStopping(patience=5, delta=0.1, restore_weights=True)
model.compile(epochs=10, learning_rate=1e-3, loss='categorical_cross_entropy', optimizer='nesterov',earlystop=stop)
model.fit(x_train, y_train, x_test, y_test, batch_size=64)