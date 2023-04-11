import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.io as sio
from Neural_Network_Base import NeuralNetwork_Base as NeuralNetwork
# from Neural_Network_Base import NeuralNetwork_Four as NeuralNetwork # To test four layer perceptron
np.random.seed(1)

'''
This file contains the algorithm used for the self-coded 3-layer and 4-layer perceptron

To run each perceptron, the import statement is changed to import NeuralNetwork_Base or NeuralNetwork_Four
The layer_numbers array is changed to account for this, adding or removing and extra hidden layer as needed
'''

# Load file
file_name = 'CMP_Data.mat'
mat_contents = sio.loadmat(file_name)
p = mat_contents["Data1"]
sio.savemat(file_name, {'Data1': p})

# check if nan entries and delete if so
if np.sum(np.isnan(p)) > 0:
    p = p[~np.isnan(p)]

# Check size of array
print(np.shape(p), '\n')

# prepare variables and target
X = p[:, 13:16]  # usage of table columns 7-10 (6:9) and 14-17 (13:16)
Y = p[:, 25]  # MRR
print(X.shape, Y.shape)

# Data normalization
X_Norm = np.empty_like(X)
for i in range(X.shape[1]):
    data_ = X[:, i]
    X_Norm[:, i] = (data_ - np.amin(data_)) / (np.amax(data_) - np.amin(data_))  # (data-min)/(max-min) to range[0 1]

# normalize Y data
Y_Min = np.amin(Y)
Y_Max = np.amax(Y)
Y_Norm = (Y - Y_Min) / (Y_Max - Y_Min)

# split training and testing data

index = np.arange(len(Y_Norm))
np.random.shuffle(index)  # disorder the original data

m = np.ceil(0.7 * len(Y))  # 70% for training and 30% for testing
m = int(m)  # covert float type to int type
X_Train = X_Norm[index[:m]]
Y_Train = Y_Norm[index[:m]]

X_Test = X_Norm[index[m:]]
Y_Test = Y_Norm[index[m:]]

layer_numbers = [3, 1, 1] # Add additional number for four layer
learning_rate = 0.01
epochs = 2000
Y_Train = np.reshape(Y_Train, (len(Y_Train), 1))
Net = NeuralNetwork(X_Train, Y_Train, layer_numbers, learning_rate, epochs)  # define an object belonging to the class
Net.train()
plt.figure()
plt.plot(Net.epoch, Net.error_history)
plt.show()

# testing
y_predict = Net.predict(X_Test)
y_predicted = y_predict * (Y_Max - Y_Min) + Y_Min
Y_Test = Y_Test * (Y_Max - Y_Min) + Y_Min
Y_Test = Y_Test.reshape(len(Y_Test), 1)
plt.scatter(y_predicted, Y_Test, c='b', marker='o')
plt.xlim(Y_Min, Y_Max)
plt.ylim(Y_Min, Y_Max)
plt.plot([Y_Min, Y_Max], [Y_Min, Y_Max], 'k-')
plt.show()


# performance evaluation
def r2(y_predicted, y):
    sst = np.sum((y - y.mean()) ** 2)
    ssr = np.sum((y_predicted - y) ** 2)
    r2 = 1 - (ssr / sst)
    return (r2)


print(f"R2: {r2(y_predicted, Y_Test)}")


def RMSE(y_predicted, y):
    rmse = np.sqrt(np.mean((y_predicted - y) ** 2))
    return rmse


print(f"RMSE: {RMSE(y_predicted, Y_Test)}")

