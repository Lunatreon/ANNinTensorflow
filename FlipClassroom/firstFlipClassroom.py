import math
from typing import Any
import numpy as np

def sigmoid(x):
    return (1/(1 + math.exp(-x)))

class Perceptron():
    #datatyps
    def __init__(self, number_inputs, use_bias):
        self.number_inputs = number_inputs
        self.bias = use_bias
        self.weights = [0.] * self.number_inputs 
        self.bias = 0.

    def set_initial_weights(self, weights, bias):
        self.weight = weights
        self.bias = bias

    def call(self, x):
        y = 0.
        for i, w in zip(x, self.weights):
            y = y + i*w
        if self.use_bias:
            y = y + self.bias
        y = sigmoid(y)    
        return y
    
def sigmoid_np(x):
    return (1/(1+np.exp(-x)))

class MLP_layer():
    def __init__(self, num_input, num_units):
        self.num_inputs = num_input
        self.num_units = num_units
        self.weights = np.zeros((num_units,num_input))
        self.bias = np.zeros((num_units,))
            
    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def set_weights_singlePerceptron(self, weights, bias, perceptron):
        self.weights[perceptron] = weights
        self.bias[perceptron] = bias

    def call(self, x):
        #x.shape: (num_input,1)
        pre_activation = self.weights @ x + np.transpose(self.bias)
        activations = sigmoid_np(pre_activation)
        return activations
