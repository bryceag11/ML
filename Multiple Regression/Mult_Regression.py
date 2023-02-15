import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.decomposition import PCA
np.random.seed(0)


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

# Prepare variables and targets
x = p[:, 6:9]  # usage of backing film, dresser, polishing table, and dresser table
y = p[:, 25]  # MRR
# x2 = p[:, 17:19] # Slurry flow of line A, B, C
print(x.shape, y.shape)

# Data normalization
for i in range(x.shape[1]):
    data_ = x[:, i]
    x[:, i] = (data_ - np.amin(data_)) / (np.amax(data_) - np.amin(data_))  # (data-min)/(max-min) to range[0 1]

# Principle component analysis
# pca = PCA(n_components=1)
# pcs = pca.fit_transform(x)
# x = pcs

const = np.ones((len(x), 1))  # constant column
x = np.concatenate((x, const), axis=1)  # group normalized input columns and constant column together
print(x.shape)

# split training and testing data subsets
# shuffle index, so that the training scenarios and testing scenarios are the same
# 70% training and 30% testing
index = np.arange(len(x))
np.random.shuffle(index)
break_point = int(np.floor(0.7 * len(index)))  # np.ceil
print(break_point)

X_Train = x[index[0:break_point], :]
Y_Train = y[index[0:break_point]]
X_Test = x[index[break_point:], :]
Y_Test = y[index[break_point:]]
print(X_Train.shape, X_Test.shape, Y_Train.shape, Y_Test.shape)


# cost function
def cost_function(x, y, B):  # B are the coefficients in MR
    J = np.sum((x.dot(B) - y) ** 2) / (2 * len(x))
    return J


# gradient descent
def gradient_descent(x, y, B, alpha, Iterations):  # alpha: learning rate

    loss_history = np.zeros(Iterations)

    for i in range(Iterations):
        gradient = x.T.dot(x.dot(B) - y) / len(x)
        B = B - alpha * gradient
        loss = cost_function(x, y, B)
        loss_history[i] = loss

    return B, loss_history

# training
B = np.random.random(X_Train.shape[1])
alpha = 0.01  # try 0.1,0.01
Iterations = 1000
newB, loss_history = gradient_descent(X_Train, Y_Train, B, alpha, Iterations)

plt.plot(np.arange(Iterations), loss_history)
print(newB.shape)

# testing
Y_predicted = X_Test.dot(newB)
plt.scatter(Y_Test, Y_predicted)


# performance evaluation, for regression problems, we need R2 and RMSE
def RMSE(y, Y_pred):
    RMSE = np.sqrt(np.mean((y - Y_pred) ** 2))
    return RMSE


def R2(y, Y_pred):
    sst = np.sum((y - y.mean()) ** 2)
    ssr = np.sum((y - Y_pred) ** 2)
    R2 = 1 - (ssr / sst)
    return R2

print(RMSE(Y_Test, Y_predicted))
print(R2(Y_Test, Y_predicted), '\n\n')

linear = linear_model.LinearRegression()  # create an object of multiple regression model
linear.fit(X_Train, Y_Train)  # training process
Y_predicted_1 = linear.predict(X_Test)
plt.scatter(Y_Test, Y_predicted_1)
print(RMSE(Y_Test, Y_predicted_1))
print(R2(Y_Test, Y_predicted_1))
