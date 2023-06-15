from layers.fullyconnected import FC
from activations import ReLU, LinearActivation, Sigmoid, Tanh
from optimizers.adam import Adam
from optimizers.gradientdescent import GD
from losses.meansquarederror import MeanSquaredError
from model import Model
import numpy as np
import pandas as pd

train_data = pd.read_csv('datasets/california_houses_price/california_housing_train.csv')
test_data = pd.read_csv('datasets/california_houses_price/california_housing_test.csv')

x_train = train_data[train_data.columns[0:8]].to_numpy()
y_train = train_data[train_data.columns[:-1]].to_numpy()

x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)

y_train = y_train.reshape(-1, 1)

models_arch = {
    "FC1": FC(8, 80, "FC1"),
    "SIGMOID1": Sigmoid(),
    "FC2": FC(80, 20, "FC2"),
    "SIGMOID2": Sigmoid(),
    "FC3": FC(20, 1, "FC3"),
    "RELU2": ReLU(),
}

model = Model(models_arch, MeanSquaredError(), GD(models_arch, learning_rate=.1))
model.train(x_train.T, y_train.T, epochs=1500, batch_size=100, verbose=10, shuffling=True)
model.save(name='./model_task_1')

columns = test_data.columns[0:8]
X_test = test_data[columns].values
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Extract the y data from all columns except the last one which contains the labels
Y_t = train_data.iloc[:, :-1].values
Y_t = Y_t.reshape(-1, 1)



y_prediction = model.predict(X_test[:, :].reshape(8, -1))
print(y_prediction[-1])
