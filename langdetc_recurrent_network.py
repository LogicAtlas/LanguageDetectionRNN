
"""
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library
"""
import AlphaBase
import os
import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import LanguageSource as LanguageSource
import LangTestData as langTestData

# Get training data
lang_data_dir = '/home/frank/data/LanguageDetectionModel/exp_data_test'
alpha_file_name = 'alpha_dog.pk'

if os.path.isfile(alpha_file_name):
    alpha_set = AlphaBase.AlphaBase.load_object_from_file(alpha_file_name)
else:
    alpha_set = AlphaBase.AlphaBase()
    alpha_set.start(lang_data_dir, 10000000)
    alpha_set.compress(0.999)
    alpha_set.save_object_to_file(alpha_file_name)
print('alpha size:', alpha_set.alpha_size, 'alpha compressed size', alpha_set.alpha_compressed_size)

lang_data = LanguageSource.LanguageSource(alpha_set)
lang_data.begin(lang_data_dir)

# Parameters
learning_rate = 0.00001
training_cycles = 100000000
batch_size = 128
display_step = 10

# Network Parameters
# number of characters in the set of languages, also the size of the one-hot vector encoding characters
n_input = alpha_set.alpha_compressed_size
n_steps = 64  # time steps
n_hidden = 512  # hidden layer num of features
n_classes = 21  # total number of class, (the number of languages in the database)

# Get test data
lang_db_test = langTestData.LangTestData()
x_test, y_test = lang_db_test.read_data('/home/frank/data/LanguageDetectionModel/europarl.test', n_steps)
test_data = lang_data.get_ml_data_matrix(n_input, x_test)  # get one-hot version of data
y_test2 = [lang_data.language_name_to_index[y_l] for y_l in y_test]  # convert the language names to indexes
test_label = lang_data.get_class_rep(y_test2, n_classes)  # convert the class indexes to one-hot vectors
test_len = len(y_test)  # get the number of test strings

# tf Graph
# input
x = tf.placeholder("float", [n_steps, None, n_input])
# desired output
y = tf.placeholder("float", [None, n_classes])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
i_state = tf.placeholder("float", [None, 2 * n_hidden])


# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def lm_rnn(_x, _i_state, _weights, _biases):
    # Reformat _x from [ ] to [n_steps*batch_size x n_input]
    xin = tf.reshape(_x, [-1, n_input])

    # Linear activation
    xin = tf.matmul(xin, _weights['hidden']) + _biases['hidden']

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    xin = tf.split(0, n_steps, xin)  # n_steps * (batch_size, n_hidden)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.95)
    # lstm_cell = rnn_cell.LSTMCell(n_hidden, use_peepholes=True, cell_clip=2, forget_bias=0.99)

    # Get lstm cell output
    # outputs - a list of n_step matrix of shape [? x n_hidden]
    # states - a list of n_step vectors of size [2*n_hidden]
    outputs, states = tf.nn.rnn(lstm_cell, xin, initial_state=_i_state)

    # Linear activation
    # Get inner loop last output
    logits = tf.matmul(outputs[-1], _weights['out']) + _biases['out']

    return logits

predictions = lm_rnn(x, i_state, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, y))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

# Evaluate model
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_cycles:
        # batch_xs - list (of length batch_size) of strings each of length n_input,
        # batch_ys - list (of length batch_size) of language ids (strings, example: 'bg','ep',...)
        batch_xs, batch_ys = lang_data.get_next_batch_one_hot(batch_size, n_steps)

        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                       i_state: np.zeros((batch_size, 2 * n_hidden))})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                i_state: np.zeros((batch_size, 2 * n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             i_state: np.zeros((batch_size, 2 * n_hidden))})
            # Read a new batch for validation
            batch_xs, batch_ys = lang_data.get_next_batch_one_hot(batch_size, n_steps)

            acc_val = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                    i_state: np.zeros((batch_size, 2 * n_hidden))})
            # Calculate batch loss
            loss_val = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                                 i_state: np.zeros((batch_size, 2 * n_hidden))})

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) +
                  ", Training Accuracy= " + "{:.5f}".format(acc) + ", Validation Loss=" +
                  "{:.6f}".format(loss_val) + ", Validation Accuracy= " + "{:.5f}".format(acc_val))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    """
    acc_sum = 0.0
    for i in range(test_len):
        test_string = x_test_nat[i]
        test_x_mat = lang_data.get_ml_data_matrix_3([test_string])
        y_t = lang_data.language_name_to_index[y_test[i]]
        test_y_mat = lang_data.get_class_rep([y_t], n_classes)
        acc = sess.run(accuracy, feed_dict={x: test_x_mat, y: test_y_mat,
                                            i_state: np.zeros((1, 2 * n_hidden))})
        acc_sum += acc
        print(i, acc, acc_sum)
    """
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             i_state: np.zeros((test_len, 2 * n_hidden))}))
