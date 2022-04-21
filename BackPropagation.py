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

    def activation_function(self, activationfn, net):
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

    def initializeNetwork(self, n_inputs):
        # num of hidden
        # list number of neurons in each hidden
        Hidden_Layers = list()
        Hidden_Layers.append([0] * 4 )# Samples

        for hiddenlayer in range(self.hidden_layers):
            neruro = [0] * self.neurons[hiddenlayer]
            Hidden_Layers.append(neruro)

        Hidden_Layers.append([0] * 3) # Output
        print("hidden layers", Hidden_Layers)
        # initialize weights
        besh = [n_inputs]
        besh.extend(self.neurons)
        besh.append(3)
        Weights = list()
        for i in range(len(besh) - 1):
            W = list()
            # Generate Random values
            for j in range(besh[i] + self.bias):
                W.append([random()] * besh[i+1])
            Weights.append(W)
        print( "weights: ", Weights)
        return Hidden_Layers, Weights

    def forward_step(self, sample, HL, Weights):

        HL[0] = sample
        # x_train with first Hidden layer
        for layer in range(1, len(HL)):
            for neuron in range(len(HL[layer])):
                for w_list in Weights[layer-1]:
                    for w_vector in w_list:
                        net = np.dot(HL[layer-1], w_vector)
                        HL[layer][neuron] = self.activation_function(self.activation_fn, net)
        print(HL)
        pass

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def htan(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def hyper_deriv (self, x):
        return  - self.htan(x) * self.htan(x)

    def backward_step(self):

        pass

    def update_weights(self):

        pass

    def backpropagation_algorithm(self, x_train, y_train, Hidden_Layers, Weights):
        for i in range(self.epochs):
            tmp = Hidden_Layers
            for j in range(len(x_train)):
                self.forward_step(x_train.iloc[j].values, tmp, Weights)
                self.backward_step()
                self.update_weights()

        return Weights

    def run_backpropagation(self):
        # Loading data
        data = pd.read_csv('IrisData.csv')
        # Choosing classes and features
        X = data.iloc[:, 0:4]
        # Ecoding the output
        Y = pd.DataFrame({"Class": data.iloc[:, -1]})
        Y = self.label_encode(Y)
        # Initialize NN
        Hidden_Layers, Weights = self.initializeNetwork(4)
        # splitting Data
        x_train, y_train, x_test, y_test = self.SplittingData(X, Y)
        # BP Algorithm
        upadted_weights = self.backpropagation_algorithm(x_train, y_train, Hidden_Layers, Weights)
