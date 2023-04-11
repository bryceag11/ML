import numpy as np
import torch

'''
This file contains all of the classes used in this project

NeuralNetwork_Base: Class for the self-coded 3-layer perceptron
    Parameters: x, y, layer_numbers [input, hidden, output], learning_rate, epochs
NeuralNetwork_Four: Class for the self-coded 4-layer perceptron, inherits from Base class
    Parameters: same as parent class
    
Net: Class for the torch coded 3-layer perceptron, inherits from torch module
    Parameters: x, y, layer_numbers [input, hidden, output], learning_rate, epochs
Net_Four: Class for the torch coded 4-layer perceptron, inherits from Net class
    Parameters: same as parent class
'''

'''
Global functions accessible to all classes
'''


# define sigmoid function and sigmoid derivative
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def sigmoid_derivative(y):
    return y * (1 - y)


'''
3-layer self-coded perceptron class
'''


class NeuralNetwork_Base:
    def __init__(self, x, y, layer_numbers, learning_rate, epochs):  # layer_numbers = [3, 2*3+1=7,1]
        # Input Neurons, Hidden Neurons, Output Neurons
        self.output = None
        self.hidden_output = None
        self.error = None
        self.input = x
        self.y = y
        self.layer_numbers = layer_numbers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.Weights0 = np.random.rand(self.layer_numbers[0], self.layer_numbers[1])  # W0: 3*7
        self.Weights1 = np.random.rand(self.layer_numbers[1], self.layer_numbers[2])  # W1: 7*1
        # For recording and plotting the error curve
        self.epoch = []
        self.error_history = []

    def forward(self):
        self.hidden_output = sigmoid(
            np.dot(self.input, self.Weights0))  # calculate the hidden neuron values, np.dot vector based multiplication
        self.output = sigmoid(np.dot(self.hidden_output, self.Weights1))  # calculate the output neuron values

    def backpropagation(self):
        self.error = np.average(np.abs(self.y - self.output))  # sum(|Yactual-Y|)/No.(Y) 100*1
        d_Weights1 = np.dot(self.hidden_output.T, (self.output - self.y) * sigmoid_derivative(
            self.output))  # gradient for W1: H'*(Yactual-Y)*sigmoid'(Y)
        layer_error1 = np.dot((self.output - self.y) * sigmoid_derivative(self.output),
                              self.Weights1.T)  # partile derivative w.r.t H (Yactual)*NN.sigmoid'(Y)*W1'
        d_Weights0 = np.dot(self.input.T, layer_error1 * sigmoid_derivative(
            self.hidden_output))  # gradient for W0  dJ/dW0 = (dJ/dY)(dY/dH)(dH/dW0) = X'*layer_error1*NN.sigmoid'(H)

        self.Weights0 = self.Weights0 - self.learning_rate * d_Weights0  # update W0
        self.Weights1 = self.Weights1 - self.learning_rate * d_Weights1  # update W1

    def train(self):
        for epoch in range(self.epochs):
            self.forward()
            self.backpropagation()
            self.epoch.append(epoch)  # np.arrange(epochs)
            self.error_history.append(self.error)

    def predict(self, new_data):
        hidden_output = sigmoid(
            np.dot(new_data, self.Weights0))  # calculate the hidden neuron values, np.dot vector based multiplication
        output = sigmoid(np.dot(hidden_output, self.Weights1))
        return output


'''
Extension of the 3-layer self-coded perceptron class
'''


class NeuralNetwork_Four(NeuralNetwork_Base):
    def __init__(self, x, y, layer_numbers, learning_rate, epochs):  # layer_numbers = [3, 2*3+1=7,1]
        super().__init__(x, y, layer_numbers, learning_rate, epochs)
        self.hidden_output2 = None
        self.Weights3 = None
        self.error = None
        self.output = None
        self.hidden_output1 = None
        self.hidden_output0 = None
        self.Weights2 = np.random.rand(self.layer_numbers[2], self.layer_numbers[3])

    def forward(self):
        self.hidden_output0 = sigmoid(
            np.dot(self.input, self.Weights0))  # calculate the hidden neuron values, np.dot vector based multiplication
        self.hidden_output1 = sigmoid(np.dot(self.hidden_output0, self.Weights1))
        self.hidden_output2 = sigmoid(
            np.dot(self.hidden_output1, self.Weights2))  # calculate the third hidden neuron values
        self.output = sigmoid(np.dot(self.hidden_output1, self.Weights2))  # calculate the output neuron values

    def backpropagation(self):
        self.error = np.average(np.abs(self.y - self.output))  # calculate the error
        d_Weights2 = np.dot(self.hidden_output1.T,
                            (self.output - self.y) * sigmoid_derivative(self.output))  # gradient for W2, 4 x 1
        layer_error1 = np.dot((self.output - self.y) * sigmoid_derivative(self.output),
                              self.Weights2.T)  # partial derivative w.r.t the 2nd hidden layer, 100 x 4
        d_Weights1 = np.dot(self.hidden_output0.T,
                            layer_error1 * sigmoid_derivative(self.hidden_output1))  # gradient for W1, 7 x
        # Does this need to be layer_error2 as indicated in video? Does it matter?
        layer_error0 = np.dot(layer_error1 * sigmoid_derivative(self.hidden_output1),
                              self.Weights1.T)  # partial derivative w.r.t the 1st hidden layer, 100 x 7
        d_Weights0 = np.dot(self.input.T,
                            layer_error0 * sigmoid_derivative(self.hidden_output0))  # gradient for W0, 3 x 7

        self.Weights0 = self.Weights0 - self.learning_rate * d_Weights0  # update W0
        self.Weights1 = self.Weights1 - self.learning_rate * d_Weights1  # update W1
        self.Weights2 = self.Weights2 - self.learning_rate * d_Weights2  # update W2

    def predict(self, new_data):
        hidden_output1 = sigmoid(np.dot(new_data, self.Weights0))
        hidden_output2 = sigmoid(np.dot(hidden_output1, self.Weights1))
        output = sigmoid(np.dot(hidden_output2, self.Weights2))
        return output


'''
3-layer torch.nn coded perceptron class
'''


class Net(torch.nn.Module):
    def __init__(self, layer_numbers):
        super().__init__()
        self.hidden1 = torch.nn.Linear(layer_numbers[0], layer_numbers[1], bias=False)
        self.output = torch.nn.Linear(layer_numbers[1], layer_numbers[2], bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


'''
Extension of the 3-layer torch.nn coded perceptron class
'''


class Net_Four(Net):
    def __init__(self, layer_numbers):
        super().__init__(layer_numbers)
        self.hidden2 = torch.nn.Linear(layer_numbers[1], layer_numbers[2], bias=False)
        self.output = torch.nn.Linear(layer_numbers[2], layer_numbers[3], bias=False)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
