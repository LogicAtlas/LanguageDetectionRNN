"""
  A class for training and testing Recurrent Neural Network language identification models.
"""
import tensorflow as tf
import numpy as np


class LMSystem(object):

    def __init__(self, _n_input, _parms):
        self.n_input = _n_input
        self.parms = _parms
        # tf Graph
        # input
        self.x = tf.placeholder("float", [_parms.n_steps, None, self.n_input])
        # desired output
        self.y = tf.placeholder("float", [None, _parms.n_classes])
        # Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
        self.i_state = tf.placeholder("float", [None, 2 * _parms.n_hidden])

        # Define weights
        self.weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input, _parms.n_hidden])),  # Hidden layer weights
            'out': tf.Variable(tf.random_normal([_parms.n_hidden, _parms.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([_parms.n_hidden])),
            'out': tf.Variable(tf.random_normal([_parms.n_classes]))
        }

        # Define the RNN for the language model
        def lm_rnn(_x, _i_state, _weights, _biases, s_parms):
            # Reformat _x from [ ] to [n_steps*batch_size x n_input]
            xin = tf.reshape(_x, [-1, self.n_input])

            # Linear activation
            xin = tf.matmul(xin, _weights['hidden']) + _biases['hidden']

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            xin = tf.split(0, s_parms.n_steps, xin)  # n_steps * (batch_size, n_hidden)

            # Define a lstm cell with tensorflow
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(s_parms.n_hidden, forget_bias=0.95)

            # Get lstm cell output
            # outputs - a list of n_step matrix of shape [? x n_hidden]
            # states - a list of n_step vectors of size [2*n_hidden]
            outputs, states = tf.nn.rnn(lstm_cell, xin, initial_state=_i_state)

            # Linear activation
            # Get inner loop last output
            logits = tf.matmul(outputs[-1], _weights['out']) + _biases['out']

            return logits

        # predict the input x using the state and weights
        self.predictions = lm_rnn(self.x, self.i_state, self.weights, self.biases, self.parms)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.predictions, self.y))  # Softmax loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.parms.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_predictions = tf.equal(tf.argmax(self.predictions, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

        # Initializing the variables
        self.init = tf.initialize_all_variables()

    def train(self, lang_data, _model_file_name):
        """ Train the model on a data set
        :param lang_data: a LanguageSource object for the training data.
        :param _model_file_name: the base name for the output model file.
        :return: nothing
        """
        saver = tf.train.Saver()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(self.init)
            step = 1
            model_count = 0  # name models with this variable
            # Keep training until reach max iterations
            while step * self.parms.batch_size < self.parms.training_cycles:
                # batch_xs - list (of length batch_size) of strings each of length n_input,
                # batch_ys - list (of length batch_size) of language ids (strings, example: 'bg','ep',...)
                batch_xs, batch_ys = lang_data.get_next_batch_one_hot(self.parms.batch_size, self.parms.n_steps)

                # Fit training using batch data
                sess.run(self.optimizer, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                    self.i_state: np.zeros((self.parms.batch_size,
                                                                            2 * self.parms.n_hidden))})
                if step % self.parms.display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                             self.i_state: np.zeros((self.parms.batch_size,
                                                                                     2 * self.parms.n_hidden))})
                    # Calculate batch loss
                    loss = sess.run(self.cost, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                          self.i_state: np.zeros((self.parms.batch_size,
                                                                                  2 * self.parms.n_hidden))})
                    # Read a new batch for validation
                    batch_xs, batch_ys = lang_data.get_next_batch_one_hot(self.parms.batch_size, self.parms.n_steps)

                    acc_val = sess.run(self.accuracy, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                                 self.i_state: np.zeros((self.parms.batch_size,
                                                                                         2 * self.parms.n_hidden))})
                    # Calculate batch loss
                    loss_val = sess.run(self.cost, feed_dict={self.x: batch_xs, self.y: batch_ys,
                                                              self.i_state: np.zeros((self.parms.batch_size,
                                                                                      2 * self.parms.n_hidden))})
                    print("Iter " + str(step*self.parms.batch_size) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) +
                          ", Training Accuracy= " + "{:.5f}".format(acc) + ", Validation Loss=" +
                          "{:.6f}".format(loss_val) + ", Validation Accuracy= " + "{:.5f}".format(acc_val))

                    # Save model periodically
                    if step % self.parms.model_step == 0:
                        model_count += 1
                        model_save_name = _model_file_name + '_' + str(model_count)  # build file name
                        saver.save(sess, model_save_name)
                        print('model saved:', model_save_name)

                step += 1
            print("Optimization Finished!")
            saver.save(sess, _model_file_name + '_final')

    def test(self, _test_data, _test_label,  _model_file_name):
        """ Test a model on a test data.
        :param _test_data: a numpy array of test data
        :param _test_label: class labels for the test data
        :param _model_file_name: the name of the model file to test with.
        :return: nothing.
        """
        saver = tf.train.Saver()
        test_len = len(_test_label)
        # Calculate accuracy
        with tf.Session() as sess:
            saver.restore(sess, _model_file_name)
            test_batch_size = 1000
            acc_sum = 0
            b_count = int(test_len/test_batch_size)

            for i in range(b_count):
                test_batch_x = _test_data[:, i*test_batch_size:(i+1)*test_batch_size, :]
                test_batch_y = _test_label[i*test_batch_size:(i+1)*test_batch_size]

                acc_sum += sess.run(self.accuracy,
                                    feed_dict={self.x: test_batch_x, self.y: test_batch_y,
                                               self.i_state: np.zeros((test_batch_size, 2 * self.parms.n_hidden))})
            acc_avg = acc_sum / b_count
            print("Testing Accuracy:", acc_avg)
