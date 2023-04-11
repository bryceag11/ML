import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from Neural_Network_Base import Net_Four as NN
np.random.seed(1)
torch.random.manual_seed(1)

'''
This file contains the algorithm used for the torch.nn coded 3-layer and 4-layer perceptron

To run each perceptron, the import statement is changed to import Net or Net_Four and the layer_numbers array
is changed to account for this, adding or removing and extra hidden layer as needed
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
Y_Norm = Y_Norm.reshape(len(Y_Norm), 1)

# split training and testing data
index = np.arange(len(Y_Norm))
np.random.shuffle(index)  # disorder the original data

m = np.ceil(0.7 * len(Y))  # 70% for training and 30% for testing
m = int(m)  # covert float type to int type
X_Train = X_Norm[index[:m]]
Y_Train = Y_Norm[index[:m]]

X_Test = X_Norm[index[m:]]
Y_Test = Y_Norm[index[m:]]

# convert numpy array to torch tensor
X_Train_Tensor = torch.tensor(X_Train).float()
X_Test_Tensor = torch.tensor(X_Test).float()
Y_Train_Tensor = torch.tensor(Y_Train).float()
Y_Test_Tensor = torch.tensor(Y_Test).float()

layer_numbers = [3, 10, 10, 1] # Add extra hidden layer for 4-layer
epochs = 10000

# instance of parent/child class depending on import statement
net = NN(layer_numbers)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_history = np.zeros(epochs)

for epoch in range(epochs):
    # forward process
    Y_pred = net(X_Train_Tensor)

    # loss
    loss = criterion(Y_pred, Y_Train_Tensor)
    # calculate gradients in backpropagation
    optimizer.zero_grad()
    loss.backward()
    # update weights
    optimizer.step()

    loss_history[epoch] = loss

plt.plot(np.arange(epochs), loss_history)
plt.show()

# testing
y_predict = net(X_Test_Tensor)
y_predicted = y_predict.detach() * (Y_Max - Y_Min) + Y_Min
Y_Test = Y_Test_Tensor * (Y_Max - Y_Min) + Y_Min
plt.scatter(y_predicted, Y_Test, c='b', marker='o')
plt.xlim(Y_Min, Y_Max)
plt.ylim(Y_Min, Y_Max)
plt.plot([Y_Min, Y_Max], [Y_Min, Y_Max], 'k-')
plt.show()


# performance evaluation
def r2(y_pred, y):
    sst = np.sum((y - y.mean()) ** 2)
    ssr = np.sum((y_pred - y) ** 2)
    r2 = 1 - (ssr / sst)
    return (r2)


print(f"R2: {r2(y_predicted.numpy(), Y_Test.numpy())}")


def RMSE(y_pred, y):
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    return rmse


print(f"RMSE: {RMSE(y_predicted.numpy(), Y_Test.numpy())}")
