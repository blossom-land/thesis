# source code credit to: https://github.com/leilin-research/GCGRNN

import numpy as np
import pandas as pd
import pickle
import datetime
import matplotlib.pyplot as plt
from utils import normalize_adj, StandardScaler
from gcn import gcn, gcnn_ddgf
import tensorflow as tf
#
# from codecarbon import EmissionsTracker
# tracker = EmissionsTracker()

# tf.compat.v1.enable_eager_execution()

# utils.py

import tensorflow as tf
import numpy as np


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    # adj[np.isnan(adj)] = 0.
    adj = tf.abs(adj)
    rowsum = tf.reduce_sum(adj, 1)  # sum of each row

    d_inv_sqrt = tf.pow(rowsum, -0.5)

    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = tf.diag(d_inv_sqrt) #depreicated https://github.com/tensorflow/tensorflow/blob/3d782b7d47b1bf2ed32bd4a246d6d6cadc4c903d/tensorflow/tools/compatibility/renames_v2.py
    d_mat_inv_sqrt = tf.linalg.diag(d_inv_sqrt)
    # d_mat_inv_sqrt = tf.linalg.tensor_diag(d_inv_sqrt)
    return tf.matmul(tf.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


# Import Data
file_Name = "/NYC_Citi_bike/NYCBikeHourly272.pickle"
fileObject = open(file_Name, 'rb')
hourly_bike = pickle.load(fileObject)
hourly_bike = pd.DataFrame(hourly_bike)
node_num = 272  # node number
#
# hourly_bike = hourly_bike.loc[:, [184, 158, 12, 88, 222, 266, 37, 38, 62, 13]]  # TOp10 station
# node_num = 10  # node number

# hourly_bike = hourly_bike.loc[:,[184, 158, 12, 88, 222, 266, 37, 38, 62, 13,  142,  39,  238,  219,  210,  54,  211,  51,  60,  218,  230,  235,  57,  0,  30]] #TOp25 station
# node_num = 25  # node number

# hourly_bike = hourly_bike.loc[:,[184,  158,  12,  88,  222,  266,  37,  38,  62,  13,  142,  39,  238,  219,  210,  54,  211,  51,  60,  218,  230,  235,  57,  0,  30,  223,  163,  225,  217,  159,  102,  196,  56,  138,  237,  231,  50,  6,  29,  105,  17,  167,  257,  11,  236,  228,  7,  185,  145,  229]]  #TOp50 station
# node_num = 50  # node number

# hourly_bike = hourly_bike.loc[:,[184,  158,  12,  88,  222,  266,  37,  38,  62,  13,  142,  39,  238,  219,  210,  54,  211,  51,  60,  218,  230,  235,  57,  0,  30,  223,  163,  225,  217,  159,  102,  196,  56,  138,  237,  231,  50,  6,  29,  105,  17,  167,  257,  11,  236,  228,  7,  185,  145,  229,  78,  212,  127,  226,  80,  156,  86,  45,  239,  224,  203,  82,  141,  41,  263,  215,  176,  220,  18,  256,  112,  97,  84,  92,  95,  8,  46,  40,  15,  4,  100,  1,  107,  99,  21,  227,  22,  85,  101,  83,  123,  106,  240,  208,  246,  144,  79,  14,  267,  209]] #top 100 stations
# node_num = 100  # node number

# Split Data into Training, Validation and Testing

feature_in = 24  # number of features at each node, e.g., bike sharing demand from past 24 hours
horizon = 1  # the length to predict, e.g., predict the future one hour bike sharing demand

X_whole = []
Y_whole = []

x_offsets = np.sort(
    np.concatenate((np.arange(-feature_in + 1, 1, 1),))
)

y_offsets = np.sort(np.arange(1, 1 + horizon, 1))

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
num_train = 20000  # 24024 hours
num_val = 2000
num_test = 2000

X_training = X_whole[:num_train, :]
Y_training = Y_whole[:num_train, :]

# shuffle the training dataset
perm = np.arange(X_training.shape[0])
np.random.shuffle(perm)
X_training = X_training[perm]
Y_training = Y_training[perm]

X_val = X_whole[num_train:num_train + num_val, :]
Y_val = Y_whole[num_train:num_train + num_val, :]

X_test = X_whole[num_train + num_val:num_train + num_val + num_test, :]
Y_test = Y_whole[num_train + num_val:num_train + num_val + num_test, :]

scaler = StandardScaler(mean=X_training.mean(), std=X_training.std())

X_training = scaler.transform(X_training)
Y_training = scaler.transform(Y_training)

X_val = scaler.transform(X_val)
Y_val = scaler.transform(Y_val)

X_test = scaler.transform(X_test)
Y_test = scaler.transform(Y_test)

# Hyperparameters

learning_rate = 0.005  # learning rate
batchsize = 100  # batch size

hidden_num_layer = [40]  # determine the number of hidden layers and the vector length at each node of each hidden layer

early_stop_th = 20  # early stopping threshold, if validation RMSE not dropping in continuous 20 steps, break
training_epochs = 500  # total training epochs

# gcn.py

import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


def gcnn_ddgf(hidden_num_layer, node_num, feature_in, horizon, learning_rate, batch_size, early_stop_th,
              training_epochs, X_training, Y_training, X_val, Y_val, X_test, Y_test, scaler):
    # ---------------------------------------- Setting  -------------------------------- #

    n_output_vec = node_num * horizon  # length of output vector at the final layer

    early_stop_k = 0  # early stop patience
    display_step = 1  # frequency of printing results
    best_val = 10000
    traing_error = 0
    test_error = 0
    predic_res = []
    bestWeightA = {}

    tf.compat.v1.reset_default_graph()

    batch_size = batch_size
    early_stop_th = early_stop_th
    training_epochs = training_epochs

    # ---------------------------------------- tf Graph input and output  -------------------------------- #

    X = tf.compat.v1.placeholder(tf.float32, [None, node_num, feature_in], name='X')
    Y = tf.compat.v1.placeholder(tf.float32, [None, n_output_vec], name='y')

    # --------------------------- define dictionaries to store layers weight & bias  --------------------- #

    weights_hidden = {}
    biases = {}
    vec_length = feature_in

    # ------------------------------- Construct model - Convolutional Layer  -------------------------------- #

    with tf.name_scope("Graph_Convolutional"):

        weights_A = tf.Variable(tf.random.normal([node_num, node_num], stddev=0.5), name="weights_A")
        weights_hidden['h1'] = tf.Variable(tf.random.normal([vec_length, hidden_num_layer[0]], stddev=0.5),
                                           name="weights_hidden_h1")
        biases['b1'] = tf.Variable(tf.random.normal([1, hidden_num_layer[0]], stddev=0.5), name="biases_b1")


        X_input = tf.transpose(X, [1, 0, 2])

        feature_len = X_input.shape[2]

        X_input = tf.reshape(X_input, [node_num, -1])

        Adj = 0.5 * (weights_A + tf.transpose(weights_A))
        Adj = normalize_adj(Adj)

        Z = tf.matmul(Adj, X_input)
        Z = tf.reshape(Z, [-1, int(feature_len)])

        signal_output = tf.add(tf.matmul(Z, weights_hidden['h1']), biases['b1'])
        signal_output = tf.nn.relu(signal_output)

    # -------------------------------- Construct model - Output Layer --------------------------- #

    with tf.name_scope("Feedforward"):

        weights_hidden['out'] = tf.Variable(tf.random.normal([hidden_num_layer[-1], horizon], stddev=0.5),
                                            name="weights_hidden_out")
        biases['bout'] = tf.Variable(tf.random.normal([1, horizon], stddev=0.5), name="biases_out")

        X_input = signal_output
        feature_len = X_input.shape[1]

        predict = tf.add(tf.matmul(signal_output, weights_hidden['out']), biases['bout'])
        predict = tf.reshape(predict, [node_num, -1, horizon])
        predict = tf.transpose(predict, [1, 0, 2])
        predict = tf.reshape(predict, [-1, node_num * horizon])

    pred = scaler.inverse_transform(predict)
    Y_original = scaler.inverse_transform(Y)

    # -------------------------------- Loss Function  --------------------------------------------------- #
    with tf.name_scope("cost"):
        #cost = tf.sqrt(tf.reduce_mean(tf.pow(pred - Y_original, 2)))
        cost = tf.sqrt(tf.reduce_mean(tf.square((Y_original - pred))))

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        #https://wizardforcel.gitbooks.io/tensorflow-tutorials-nlintz/content/09_tensorboard.html
        # Add scalar summary for cost
        tf.summary.scalar("cost", cost)

    # ------------------------------------- Initializing the variables  -------------------------------- #
    tvar = tf.compat.v1.trainable_variables()
    tvar_name = [x.name for x in tvar]
    print(tvar)

    #https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model/38161314
    tot= np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
    print("total",tot)

    saver = tf.compat.v1.train.Saver()
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:

        # tracker.start()
        graph = tf.compat.v1.get_default_graph()
        sess.run(init)

        table_trainloss = []
        table_valloss = []

        # #https://github.com/tensorflow/tensorflow/issues/32809
        #
        # run_meta = tf.compat.v1.RunMetadata()
        # opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        #
        # # We use the Keras session graph in the call to the profiler.
        # flops = tf.compat.v1.profiler.profile(graph=graph,
        #                                       run_meta=run_meta, cmd='op', options=opts)

        #return flops.total_float_ops

        for epoch in range(training_epochs):



            avg_cost = 0.
            num_train = X_training.shape[0]
            total_batch = int(num_train / batch_size)

            for i in range(total_batch):
                _, c = sess.run([optimizer, cost], feed_dict={X: X_training[i * batch_size:(i + 1) * batch_size, ],
                                                              Y: Y_training[i * batch_size:(i + 1) * batch_size, ]})

                avg_cost += c * batch_size  # / total_batch

            # rest part of training dataset
            if total_batch * batch_size != num_train:
                _, c = sess.run([optimizer, cost], feed_dict={X: X_training[total_batch * batch_size:num_train, ],
                                                              Y: Y_training[total_batch * batch_size:num_train, ]})
                avg_cost += c * (num_train - total_batch * batch_size)

            avg_cost = avg_cost / num_train
        #
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "Training loss" + " =", "{:.9f}".format(avg_cost))
                table_trainloss.append(avg_cost)

            # validation
            c_val = sess.run([cost], feed_dict={X: X_val, Y: Y_val})
            c_val = c_val[0]

            print("Validation loss" + ":", c_val)

            table_valloss.append(c_val)

            # testing
            c_tes, preds, Y_true, weights_A_final = sess.run([cost, pred, Y_original, weights_A],
                                                             feed_dict={X: X_test, Y: Y_test})

            if c_val < best_val:
                best_val = c_val
                # save model
                saver.save(sess, "model")
                test_error = c_tes
                traing_error = avg_cost
                predic_res = preds
                bestWeightA = weights_A_final
                early_stop_k = 0  # reset to 0

            # update early stopping patience
            if c_val >= best_val:
                early_stop_k += 1

            # threshold
            if early_stop_k == early_stop_th:
                break

        print("epoch is ", epoch)
        print("training loss " + " is ", traing_error)
        print("Optimization Finished! the lowest validation loss" + " is ", best_val)
        print("The test loss " + " is ", test_error)

    # sanity check to compare tft - RMSE & MAE on test set
    from sklearn.metrics import mean_absolute_error as mae
    mae = mae(Y_true, predic_res)
    print("Test MAE is", mae)

    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(Y_true, predic_res, squared=False)
    print("Test RMSE is", rmse)

    # emission: float = tracker.stop()

    return best_val, predic_res, Y_true, test_error, bestWeightA, table_trainloss, table_valloss   \
        # , emission
#

# Training

start_time = datetime.datetime.now()



val_error, predic_res, test_Y, test_error, bestWeightA, table_trainloss, table_valloss = gcnn_ddgf(hidden_num_layer,
                                                                                                   node_num, feature_in,
                                                                                                   horizon,
                                                                                                   learning_rate,
                                                                                                   batchsize,
                                                                                                   early_stop_th,
                                                                                                   training_epochs,
                                                                                                   X_training,
                                                                                                   Y_training, X_val,
                                                                                                   Y_val, X_test,
                                                                                                   Y_test, scaler)
# , emission_s
# print(f"Emissions: {emission_s} kg")


end_time = datetime.datetime.now()
print('Total training time: ', end_time - start_time)
#
np.savetxt("prediction_1hours.csv", predic_res, delimiter = ',')
np.savetxt("prediction_Y_1hours.csv", test_Y, delimiter = ',')
# Save the adjacency matrix for analysis
np.savetxt("bestWeight.csv", bestWeightA, delimiter=',')
