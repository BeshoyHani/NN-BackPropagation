class BackPropagation:
    def __init__(self, hidden_layers, neurons, eta, epochs, bias, activation_fn):
        self.hidden_layers = hidden_layers # integer value
        self.neurons = neurons # integer value
        self.eta = eta # Float value
        self.epochs = epochs # integer
        self.bias = bias # integer whether 0 => Don't add bias or 1 => Add bias
        self.activation_fn = activation_fn # Text containing Name of the Activation Function