from random import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BackPropagation:
    def __init__(self, hidden_layers, neurons, eta, epochs, bias, activation_fn):
        self.hidden_layers = hidden_layers  # integer value
        self.neurons = neurons  # List of integers represent #neurons per each layer
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
            if Y.iloc[i, 0] == "Iris-setosa":
                Y.iloc[i, 0] = 1
            elif Y.iloc[i, 0] == "Iris-versicolor":
                Y.iloc[i, 0] = 2
            elif Y.iloc[i, 0] == "Iris-virginica":
                Y.iloc[i, 0] = 3
        return Y

    def initializeNetwork(self, n_inputs):
        # num of hidden
        # list number of neurons in each hidden
        Hidden_Layers = list()
        for hiddenlayer in range(self.hidden_layers):
            neruro = [0] * self.neurons[hiddenlayer]
            Hidden_Layers.append(neruro)
        # initialize weights
        besh = [n_inputs]
        besh.extend(self.neurons)
        besh.append(3)
        Weights = list()
        for i in range(len(besh) - 1):
            W = list()
            # Generate Random values
            for j in range(((besh[i] + self.bias) * besh[i + 1])):
                W.append(random())
            Weights.append(W)
        return Hidden_Layers, Weights

    def backpropagation_algorithm(self, x_train, y_train):

        return

    def SplittingData(self, X, Y):
        x_train, y_train, x_test, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in range(3):
            X_trainC1, X_testC1, y_trainC1, y_testC1 = train_test_split(X[0 + i * 50:50 + i * 50],
                                                                        Y[i * 50:50 + i * 50], test_size=0.40,
                                                                        shuffle=False)

            x_train = x_train.append(X_trainC1, ignore_index=True)
            y_train = y_train.append(y_trainC1, ignore_index=True)
            x_test = x_test.append(X_testC1, ignore_index=True)
            y_test = y_test.append(y_testC1, ignore_index=True)

        return x_train, y_train, x_test, y_test

    def run_backpropagation(self):
        # Loading data
        data = pd.read_csv('IrisData.csv')
        # Choosing classes and features
        X = data.iloc[:, 0:4]
        # Ecoding the output
        Y = pd.DataFrame({"Class": data.iloc[:, -1]})
        Y = self.label_encode(Y)
        # Initialize NN
        Hidden_Layers, Weights = self.initializeNetwork(len(X))
        # splitting Data
        x_train, y_train, x_test, y_test = self.SplittingData(X, Y)
