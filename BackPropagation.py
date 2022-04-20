import numpy as np
import pandas as pd


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

    def label_encode(Y):
        for i in range(len(Y)):
            if Y == "Iris-setosa":
                Y[0][i] = 1
            elif Y[0][i] == "Iris-versicolor":
                Y[0][i] = 2
            elif Y[0][i] == "Iris-virginica":
                Y[0][i] = 3
        return Y

    def run_backpropagation(self):
        # Loading data
        data = pd.read_csv('IrisData.csv')
        # Choosing classes and features
        X = data.iloc[:, 0:4]
        # Ecoding the output
        Y = data.iloc[:, -1]
        Y = self.label_encode(Y)
        # splitting Data
        # X_train, Y_train, X_test, Y_test = self.SplittingData(X, Y)

