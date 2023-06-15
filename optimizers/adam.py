import numpy as np

from layers.convolution2d import Conv2D
from layers.fullyconnected import FC
# TODO: Implement Adam optimizer
class Adam:
    def __init__(self, layers_list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-4):
        self.layers = layers_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        for layer in layers_list:
            # TODO: Initialize V and S for each layer (v and s are lists of zeros with the same shape as the parameters)
            if type(layers_list[layer]) == FC or type(layers_list[layer]) == Conv2D:
                v = [np.zeros_like(p) for p in layers_list[layer].parameters]
                s = [np.zeros_like(p) for p in layers_list[layer].parameters]
                self.V[layer] = v
                self.S[layer] = s

    def update(self, grads, name, epoch=1):
        layer = self.layers[name]
        params = []
        # TODO: Implement Adam update
        for index in range(len(grads)):
            self.V[name][index] = self.beta1 * self.V[name][index] + (1 - self.beta1) * grads[index]
            self.S[name][index] = self.beta2 * self.S[name][index] + (1 - self.beta2) * np.square(grads[index])
            self.V[name][index] /= (1 - np.power(self.beta1, epoch))  # Corrected V
            self.S[name][index] /= (1 - np.power(self.beta2, epoch))  # Corrected S
            params.append(layer.parameters[index] - self.learning_rate * (
                    self.V[name][index] / (np.sqrt(self.S[name][index]) + self.epsilon)))
        return params
