
#source code credit to: https://github.com/leilin-research/GCGRNN

import numpy as np
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt
from utils import normalize_adj, StandardScaler
from gcn import gcn, gcnn_ddgf
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()

# Import Data
file_Name = "/NYC_Citi_bike/NYCBikeHourly272.pickle"
fileObject = open(file_Name,'rb') 
hourly_bike = pickle.load(fileObject) 
hourly_bike = pd.DataFrame(hourly_bike)
# hourly_bike = hourly_bike.loc[:,[184, 158, 12, 88, 222, 266, 37, 38, 62, 13]] #TOp10 station
# hourly_bike = hourly_bike.loc[:,[184, 158, 12, 88, 222, 266, 37, 38, 62, 13,  142,  39,  238,  219,  210,  54,  211,  51,  60,  218,  230,  235,  57,  0,  30]] #TOp25 station
# hourly_bike = hourly_bike.loc[:,[184,  158,  12,  88,  222,  266,  37,  38,  62,  13,  142,  39,  238,  219,  210,  54,  211,  51,  60,  218,  230,  235,  57,  0,  30,  223,  163,  225,  217,  159,  102,  196,  56,  138,  237,  231,  50,  6,  29,  105,  17,  167,  257,  11,  236,  228,  7,  185,  145,  229]]  #TOp50 station
# hourly_bike = hourly_bike.loc[:,[184,  158,  12,  88,  222,  266,  37,  38,  62,  13,  142,  39,  238,  219,  210,  54,  211,  51,  60,  218,  230,  235,  57,  0,  30,  223,  163,  225,  217,  159,  102,  196,  56,  138,  237,  231,  50,  6,  29,  105,  17,  167,  257,  11,  236,  228,  7,  185,  145,  229,  78,  212,  127,  226,  80,  156,  86,  45,  239,  224,  203,  82,  141,  41,  263,  215,  176,  220,  18,  256,  112,  97,  84,  92,  95,  8,  46,  40,  15,  4,  100,  1,  107,  99,  21,  227,  22,  85,  101,  83,  123,  106,  240,  208,  246,  144,  79,  14,  267,  209]] #top 100 stations

# Split Data into Training, Validation and Testing

node_num = 272 # node number
feature_in = 24 # number of features at each node, e.g., bike sharing demand from past 24 hours
horizon = 1 # the length to predict, e.g., predict the future one hour bike sharing demand

X_whole = []
Y_whole = []

x_offsets = np.sort(
    np.concatenate((np.arange(-feature_in+1, 1, 1),))
)

y_offsets = np.sort(np.arange(1, 1+ horizon, 1))

min_t = abs(min(x_offsets))
max_t = abs(hourly_bike.shape[0] - abs(max(y_offsets)))  # Exclusive
for t in range(min_t, max_t):
    x_t = hourly_bike.iloc[t + x_offsets, 0:node_num].values.flatten('F')
    y_t = hourly_bike.iloc[t + y_offsets, 0:node_num].values.flatten('F')
    X_whole.append(x_t)
    Y_whole.append(y_t)

X_whole = np.stack(X_whole, axis=0)
Y_whole = np.stack(Y_whole, axis=0)


X_whole = np.reshape(X_whole, [X_whole.shape[0], node_num, feature_in])
num_samples = X_whole.shape[0]
# num_train = X_whole.shape[0]-4000 #complete dataset
num_train = 20000 # 24024 hours
num_val = 2000
num_test = 2000

X_training = X_whole[:num_train, :]
Y_training = Y_whole[:num_train, :]

# shuffle the training dataset
perm = np.arange(X_training.shape[0])
np.random.shuffle(perm)
X_training = X_training[perm]
Y_training = Y_training[perm]

X_val = X_whole[num_train:num_train+num_val, :]
Y_val = Y_whole[num_train:num_train+num_val, :]

X_test = X_whole[num_train+num_val:num_train+num_val+num_test, :]
Y_test = Y_whole[num_train+num_val:num_train+num_val+num_test, :]

scaler = StandardScaler(mean=X_training.mean(), std=X_training.std())

X_training = scaler.transform(X_training)
Y_training = scaler.transform(Y_training)

X_val = scaler.transform(X_val)
Y_val = scaler.transform(Y_val)

X_test = scaler.transform(X_test)
Y_test = scaler.transform(Y_test)

# Hyperparameters

learning_rate = 0.005 # learning rate
decay = 0.9
batchsize = 128 # batch size

hidden_num_layer = [64] # determine the number of hidden layers and the vector length at each node of each hidden layer
reg_weight = [0] # regularization weights for adjacency matrices L1 loss

keep = 1 # drop out probability

early_stop_th = 20 # early stopping threshold, if validation RMSE not dropping in continuous 20 steps, break
training_epochs = 500 # total training epochs

# Training

start_time = datetime.datetime.now()
val_error, predic_res, test_Y, test_error, bestWeightA , table_trainloss, table_valloss= gcnn_ddgf(hidden_num_layer, reg_weight, node_num, feature_in, horizon, learning_rate, decay, batchsize, keep, early_stop_th, training_epochs, X_training, Y_training, X_val, Y_val, X_test, Y_test, scaler,'RMSE')
end_time = datetime.datetime.now()

print('Total training time: ', end_time-start_time)
#
np.savetxt("Top100-prediction_1hours.csv", predic_res, delimiter = ',')
np.savetxt("Top100-prediction_Y_1hours.csv", test_Y, delimiter = ',')

#
# credit:  https://stackoverflow.com/questions/54749649/how-to-plot-epoch-vs-val-acc-and-epoch-vs-val-loss-graph-in-cnn

# Create count of the number of epochs
epoch_count = range(1, len(table_trainloss) + 1)

# Visualize loss history
plt.plot(epoch_count, table_trainloss, 'g--')
plt.plot(epoch_count, table_valloss, 'b-')
plt.legend(['Training Loss', 'Val Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
#

# Save the adjacency matrix for analysis
np.savetxt("bestWeight.csv", bestWeightA["A0"], delimiter = ',')

# sanity check to compare tft - RMSE & MAE on test set
from sklearn.metrics import mean_absolute_error as mae
mae=mae(test_Y,predic_res)
print("test mae is", mae)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(test_Y,predic_res, squared=False)
rmse
print("test rmse is", rmse)