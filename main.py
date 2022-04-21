from tkinter import *
from tkinter.ttk import Combobox
from BackPropagation import BackPropagation


class Main:
    def __init__(self):
        self.window = Tk()
        self.window.geometry("600x300")

        # Number of Hidden Layers
        self.hidden_layers_frame = Frame()
        self.hidden_layers = self.create_textbox_frame(self.hidden_layers_frame, "Number of hidden layers: ", 5, 1, 1,
                                                       0)
        # create_text_box() takes 1) parent frame 2) label 3) initial value 4) x position 5) y position 6) padding value

        # Number of Neurons
        self.neurons_frame = Frame()
        self.neurons = self.create_textbox_frame(self.neurons_frame, "Neurons /hidden layer: ", "2, 5, 3, 4, 1", 2, 1, 7)

        # Learning Rate
        self.eta_frame = Frame()
        self.eta = self.create_textbox_frame(self.eta_frame, "Learning Rate (eta): ", 0.001, 3, 1, 12)

        # Number of Epochs
        self.epochs_frame = Frame()
        self.epochs = self.create_textbox_frame(self.epochs_frame, "Epochs (m): ", 1000, 4, 1, 33)

        # add bias or not
        self.addBias = IntVar()
        Checkbutton(text="Add Bias", variable=self.addBias).grid(row=3, column=2, pady=25)

        # Activation function
        self.activation_frame = Frame()
        self.activation_cb = self.create_combobox(self.activation_frame, "Activation Function: ", 2, 2)
        self.activation_cb.current(0)

        Button(text="OK", width=10, bg='white', command=self.submit).grid(row=4, column=2, padx=0)
        self.window.mainloop()

    def create_textbox_frame(self, parent, label, initial_value, x, y, padding_x):
        Label(parent, text=label, font='Calibre 10').grid(row=1, column=1, padx=10 + padding_x)
        textbox = Entry(parent)
        textbox.grid(row=1, column=2)
        textbox.insert(END, initial_value)
        parent.grid(row=x, column=y, pady=10)
        return textbox

    def create_combobox(self, parent, label, x, y):
        Label(parent, text=label, font='Calibre 10').grid(row=1, column=1, padx=10)
        selected_language = StringVar()
        activation_cb = Combobox(parent, textvariable=selected_language, justify="center")
        activation_cb['values'] = ("Sigmoid", "Hyperbolic Tangent")
        activation_cb['state'] = 'readonly'
        activation_cb.grid(row=1, column=2)
        parent.grid(row=x, column=y, pady=10)
        return activation_cb

    def submit(self):
        hidden_layers = int(self.hidden_layers.get())

        neurons = self.neurons.get()
        neurons = neurons.split(',')
        neurons = list(map(int, neurons))

        eta = float(self.eta.get())
        epochs = int(self.epochs.get())
        bias = int(self.addBias.get())
        activation_fn = self.activation_cb.get()

        backPropagation = BackPropagation(hidden_layers, neurons, eta, epochs, bias, activation_fn)


main = Main()
