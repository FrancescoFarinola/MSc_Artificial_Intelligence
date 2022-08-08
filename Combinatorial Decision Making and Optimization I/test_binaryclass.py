import numpy as np
import tensorflow.keras.datasets.mnist as mnist
import neuralnet as nn

(x_train, y_train), (x_test, y_test) = mnist.load_data()

index_0 = np.where(y_train == 0)
index_8 = np.where(y_train == 8)

index = np.concatenate([index_0[0], index_8[0]])
np.random.seed(1)
np.random.shuffle(index)

train_y = y_train[index]
train_x = x_train[index]

train_y[np.where(train_y == 0)] = 0
train_y[np.where(train_y == 8)] = 1

index_0 = np.where(y_test == 0)
index_8 = np.where(y_test == 8)

index = np.concatenate([index_0[0], index_8[0]])
np.random.shuffle(index)

test_y = y_test[index]
test_x = x_test[index]

test_y[np.where(test_y == 0)] = 0
test_y[np.where(test_y == 8)] = 1

train_x = np.array(train_x / 255., dtype=np.float64)
test_x = np.array(test_x / 255., dtype=np.float64)

train_x = train_x.reshape(train_x.shape[0], -1)
test_x = test_x.reshape(test_x.shape[0], -1)

model = nn.NeuralNetwork()
model.add(nn.Layer(64, activation='relu'))
model.add(nn.Layer(16, activation='relu'))
model.add(nn.Layer(1, activation='sigmoid'))
stop = nn.EarlyStopping(patience=3, delta=0.01, restore_weights=True)
model.compile(epochs=20, learning_rate=1e-3, loss='binary_cross_entropy', optimizer='momentum',earlystop=stop)
model.fit(train_x, train_y, test_x, test_y, batch_size=64)