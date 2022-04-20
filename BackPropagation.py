import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BackPropagation:
    def __init__(self, hidden_layers, neurons, eta, epochs, bias, activation_fn):
        self.hidden_layers = hidden_layers  # integer value
        self.neurons = neurons  # integer value
        self.eta = eta  # Float value
        self.epochs = epochs  # integer
        self.bias = bias  # integer whether 0 => Don't add bias or 1 => Add bias
        self.activation_fn = activation_fn  # Text containing Name of the Activation Function
        self.run_backpropagation()

    def activation_function(activationfn, net):
        if activationfn == "Sigmoid":
            return 1 / (1 + np.exp(-net))
        elif activationfn == "Hyperbolic Tangent":
            return (np.exp(net) - np.exp(-net)) / (np.exp(net) + np.exp(-net))

    def label_encode(self, Y):
        for i in range(len(Y)):
            if Y.iloc[i] == "Iris-setosa":
                Y.iloc[i] = 1
            elif Y.iloc[i] == "Iris-versicolor":
                Y.iloc[i] = 2
            elif Y.iloc[i] == "Iris-virginica":
                Y.iloc[i] = 3
        return Y
    def initializeNetwork(self,n_inputs):
        # num of hidden
        # list number of neurons in each hidden
        Hidden_Layers = list()
        for hiddenlayer in range(self.hidden_layers):
            neruro= [0] * self.neurons[hiddenlayer]
            Hidden_Layers.append(neruro)
        # initialize weights
        Weights = list()

        return Hidden_Layers,Weights

    def backpropagation_algorithm(self, x_train, y_train):

        return

    def run_backpropagation(self):
        # Loading data
        data = pd.read_csv('IrisData.csv')
        # Choosing classes and features
        X = data.iloc[:, 0:4]
        # Ecoding the output
        Y = data.iloc[:, -1]
        Y = self.label_encode(Y)
        print(Y)
        # splitting Data

