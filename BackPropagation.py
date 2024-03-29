import copy
from random import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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
        res  = []
        for i in range(len(Y)):
            if Y.iloc[i, 0] == "Iris-setosa":
                res.append([1,0,0])
            elif Y.iloc[i, 0] == "Iris-versicolor":
                res.append([0,1,0])
            elif Y.iloc[i, 0] == "Iris-virginica":
                res.append([0,0,1])
        return res

    def SplittingData(self, X, Y):
        x_train, y_train, x_test, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for i in range(3):
            X_trainC1, X_testC1, y_trainC1, y_testC1 = train_test_split(X[0 + i * 50:50 + i * 50],
                                                                        Y[i * 50:50 + i * 50], test_size=0.40,
                                                                        shuffle=True)

            x_train = x_train.append(X_trainC1, ignore_index=True)
            y_train = y_train.append(y_trainC1, ignore_index=True)
            x_test = x_test.append(X_testC1, ignore_index=True)
            y_test = y_test.append(y_testC1, ignore_index=True)

        return x_train, y_train, x_test, y_test

    def initializeNetwork(self, n_inputs):
        # num of hidden
        # list number of neurons in each hidden
        Hidden_Layers = list()
        Hidden_Layers.append([0] * 4)  # Samples

        for hiddenlayer in range(self.hidden_layers):
            neruro = [0] * self.neurons[hiddenlayer]
            Hidden_Layers.append(neruro)

        Hidden_Layers.append([0] * 3)  # Output
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
                W.extend([random()] * besh[i + 1])
            Weights.append(W)
        print("weights: ", Weights)
        return Hidden_Layers, Weights

    def forward_step(self, sample, HL, Weights):

        HL[0] = list(sample)
        # x_train with first Hidden layer
        for layer in range(1, len(HL)):
            for neuron in range(len(HL[layer])):
                net = 0
                for i in range(0, len(HL[layer - 1])):
                    idx = len(HL[layer - 1]) * neuron + i
                    net += Weights[layer - 1][idx] * HL[layer - 1][i]
                HL[layer][neuron] = self.activation_function(self.activation_fn, net)
        return HL

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def htan(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def hyper_deriv(self, x):
        return - self.htan(x) * self.htan(x)

    def backward_step(self, Weights, F, y_train):
        Sigma = copy.deepcopy(F)
        n = len(F)
        for neuron in range(len(F[n - 1])):
            e = y_train[neuron] - F[n - 1][neuron]
            fd = F[n - 1][neuron] * (1 - F[n - 1][neuron])
            Sigma[n - 1][neuron] = e * fd

        for layer in reversed(range(1, n - 1)):
            for neuron in range(len(F[layer])):
                Sigma[layer][neuron] = 0
                if self.activation_fn == "Sigmoid":
                    fd = F[layer][neuron] * (1 - F[layer][neuron])
                else:
                    fd = 0
                for i in range(0, len(F[layer+1])):
                    idx = len(F[layer]) * i + neuron
                    Sigma[layer][neuron] += Weights[layer][idx] * Sigma[layer + 1][i]

                Sigma[layer][neuron] *= fd

        return Sigma

    def update_weights(self, Weights , F , S):
       # print(len(Weights),len(S))
        for layer in range(0,len(Weights)):
            for w in range(0,len(Weights[layer])):
                S_idx = w // len(F[layer])
                x_idx = w % len(F[layer])
                Weights[layer][w] += self.eta * F[layer][x_idx] * S[layer+1][S_idx]
        return Weights

    def backpropagation_algorithm(self, x_train, y_train, Hidden_Layers, Weights):
        for i in range(self.epochs):
            for j in range(len(x_train)):
                F = self.forward_step(x_train.iloc[j].values, copy.deepcopy(Hidden_Layers), Weights)
                S = self.backward_step(Weights, copy.deepcopy(F), y_train[j])
                Weights = self.update_weights(Weights,F,S)
        return Weights

    def testing (self,x_test , HL ,Weights):
        y_test = []
        HL[0] = list(x_test)
        for layer in range(1, len(HL)):
            for neuron in range(len(HL[layer])):
                net = 0
                for i in range(0, len(HL[layer - 1])):
                    idx = len(HL[layer - 1]) * neuron + i
                    net += Weights[layer - 1][idx] * HL[layer - 1][i]

                HL[layer][neuron] = self.activation_function(self.activation_fn, net)
                if layer == len(HL) - 1:
                    y_test.append(HL[layer][neuron])

        y=[]
        for i in range(len(y_test)):
            if i == y_test.index(max(y_test)) :
                y.append(1)
            else :
                y.append(0)
        return y

    def run_backpropagation(self):
        # Loading data
        data = pd.read_csv('IrisData.csv')
        # Choosing classes and features
        X = data.iloc[:, 0:4]
        # Ecoding the output
        Y = pd.DataFrame({"Class": data.iloc[:, -1]})
        # Y = self.label_encode(Y)
        # Initialize NN
        Hidden_Layers, Weights = self.initializeNetwork(4)
        # splitting Data
        x_train, y_train, x_test, y_test = self.SplittingData(X, Y)
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = self.label_encode(y_train)
        y_test = self.label_encode(y_test)
        # BP Algorithm

        upadted_weights = self.backpropagation_algorithm(x_train, y_train, Hidden_Layers, Weights)
        y_pred = []
        print(type(Hidden_Layers))
        print(type(Weights))
        print(upadted_weights)
        #print(x_test)
        for i in range(len(x_test)):
            y_pred.append(self.testing(x_test.iloc[i].values,copy.deepcopy(Hidden_Layers),upadted_weights))

        cnt = 0
        for i in range(0,len(y_pred)):
            s1 = y_pred[i].index(max(y_pred[i]))
            s2 = list(y_test[i]).index(max(list(y_test[i])))

            if s1 == s2 :
                cnt+=1

        print(cnt , len(y_pred))
        print(cnt / len(y_pred))