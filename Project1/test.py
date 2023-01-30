# I/O for matlab file read and write

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

mat_contents = sio.loadmat('CMP_Data.mat')
a = mat_contents["Data1"]
sio.savemat('CMP_Data.mat',{'Data1': a})

# check if nan entries and delete if so
if np.sum(np.isnan(a)) > 0:
    a = a[~np.isnan(a)]

# Check size of array
print(np.shape(a))

# Create new array to hold statistical analysis results
analysis = []
# start from the seventh column
new_a = a[:, 6:]

# Iterate through loop to perform analysis on each column
for i in range(new_a.shape[1]):
    col = new_a[:, i]
    a_mean = np.mean(col)
    a_max = np.max(col)
    a_min = np.min(col)
    a_std = np.std(col)
    a_ten = np.percentile(col, 10)
    a_ninety = np.percentile(col, 90)
    # add values to the results array after each iteration
    analysis.append([a_mean, a_max, a_min, a_std, a_ten, a_ninety])

# convert results to a numpy array
analysis = np.array(analysis)
print(analysis)
MRR_ten = analysis[19,4] # store the 10% of MRR values
MRR_ninety = analysis[19,5] # store the 90% of MRR values
# Identify the row indexes whose MRR are either below 10% or above 90% of MRR values
MRR_data = a[:, 25]  # Column 26 for the MRR Data

# create empty array for the indexes
indexes = []
for i in range(MRR_data.shape[0]): # length of the 1D array
    if MRR_data[i] < MRR_ten:
        indexes.append([i])
        print(f"Row index {i} is below the 10th percentile of MRR values")
    elif MRR_data[i] > MRR_ninety:
        indexes.append([i])
        print(f"Row index {i} is above the 90th percentile of MRR values")
indexes = np.array(indexes) # convert to a numpy array

# Create an array containing data from columns 7-25
plot_data = a[:,6:25]

fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(10, 20))
ax = ax.flatten()

# iterate through the number of columns
for i in range(19):
    ax[i].plot(plot_data[:, i], color='blue')
    ax[i].plot(MRR_data, color='red')
    ax[i].set_title(f"Column {i+7} vs MRR")
fig.legend(['Individual Process Parameters', "MRR Data"])
plt.show()



