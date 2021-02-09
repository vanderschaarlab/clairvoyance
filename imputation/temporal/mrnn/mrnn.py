"""MRNN core functions.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar, 
                     "Estimating Missing Data in Temporal Data Streams Using Multi-Directional Recurrent Neural Networks," 
                     in IEEE Transactions on Biomedical Engineering, vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
---------------------------------------------------
(1) Train RNN part
(2) Test RNN part
(3) Train FC part
(4) Test FC part
"""

# Necessary Packages
from tqdm import tqdm
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
import numpy as np
from imputation.temporal.mrnn.mrnn_utils import biGRUCell, initial_point_interpolation


class MRnn:
    """MRNN class with core functions.
    
    Attributes:
        - x: incomplete data
        - save_file_directory: directory for training model saving
    """

    def __init__(self, x, save_file_directory):

        # Set Parameters
        self.no, self.seq_len, self.dim = x.shape
        self.save_file_directory = save_file_directory

    def rnn_train(self, x, m, t, f, model_parameters):
        """Train RNN for each feature.
        
        Args:
            - x: incomplete data
            - m: mask matrix
            - t: time matrix
            - f: feature index
            - model_parameters:
                - h_dim: hidden state dimensions
                - batch_size: the number of samples in mini-batch
                - iteration: the number of iteration
                - learning_rate: learning rate of model training
        """
        tf.compat.v1.reset_default_graph()

        self.h_dim = model_parameters["h_dim"]
        self.batch_size = model_parameters["batch_size"]
        self.iteration = model_parameters["iteration"]
        self.learning_rate = model_parameters["learning_rate"]

        with tf.compat.v1.Session() as sess:
            # input place holders
            target = tf.placeholder(tf.float32, [self.seq_len, None, 1])
            mask = tf.placeholder(tf.float32, [self.seq_len, None, 1])

            # Build rnn object
            rnn = biGRUCell(3, self.h_dim, 1)
            outputs = rnn.get_outputs()
            loss = tf.sqrt(tf.reduce_mean(tf.square(mask * outputs - mask * target)))
            optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
            train = optimizer.minimize(loss)

            sess.run(tf.compat.v1.global_variables_initializer())

            # Training
            for i in range(self.iteration):
                # Batch selection
                batch_idx = np.random.permutation(x.shape[0])[: self.batch_size]

                temp_input = np.dstack((x[:, :, f], m[:, :, f], t[:, :, f]))
                temp_input_reverse = np.flip(temp_input, 1)

                forward_input = np.zeros([self.batch_size, self.seq_len, 3])
                forward_input[:, 1:, :] = temp_input[batch_idx, : (self.seq_len - 1), :]

                backward_input = np.zeros([self.batch_size, self.seq_len, 3])
                backward_input[:, 1:, :] = temp_input_reverse[batch_idx, : (self.seq_len - 1), :]

                _, step_loss = sess.run(
                    [train, loss],
                    feed_dict={
                        mask: np.transpose(np.dstack(m[batch_idx, :, f]), [1, 2, 0]),
                        target: np.transpose(np.dstack(x[batch_idx, :, f]), [1, 2, 0]),
                        rnn._inputs: forward_input,
                        rnn._inputs_rev: backward_input,
                    },
                )

            # Save model
            inputs = {"forward_input": rnn._inputs, "backward_input": rnn._inputs_rev}
            outputs = {"imputation": outputs}

            save_file_name = self.save_file_directory + "/rnn_feature_" + str(f + 1) + "/"
            tf.compat.v1.saved_model.simple_save(sess, save_file_name, inputs, outputs)

    def rnn_predict(self, x, m, t, median_vals):
        """Impute missing data using RNN block.
        
        Args:
            - x: incomplete data
            - m: mask matrix
            - t: time matrix
            - median_vals: median imputation for variables without observations
        
        Returns:
            - imputed_x: imputed data by rnn block
        """
        # Output Initialization
        imputed_x = np.zeros([self.no, self.seq_len, self.dim])

        # For each feature
        for f in tqdm(range(self.dim)):

            temp_input = np.dstack((x[:, :, f], m[:, :, f], t[:, :, f]))
            temp_input_reverse = np.flip(temp_input, 1)

            forward_input = np.zeros([self.no, self.seq_len, 3])
            forward_input[:, 1:, :] = temp_input[:, : (self.seq_len - 1), :]

            backward_input = np.zeros([self.no, self.seq_len, 3])
            backward_input[:, 1:, :] = temp_input_reverse[:, : (self.seq_len - 1), :]

            save_file_name = self.save_file_directory + "/rnn_feature_" + str(f + 1) + "/"

            # Load saved model
            graph = tf.Graph()
            with graph.as_default():
                with tf.compat.v1.Session() as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.SERVING], save_file_name)
                    fw_input = graph.get_tensor_by_name("inputs:0")
                    bw_input = graph.get_tensor_by_name("inputs_rev:0")
                    output = graph.get_tensor_by_name("map/TensorArrayStack/TensorArrayGatherV3:0")

                    imputed_data = sess.run(output, feed_dict={fw_input: forward_input, bw_input: backward_input})

                    imputed_x[:, :, f] = (1 - m[:, :, f]) * np.transpose(np.squeeze(imputed_data)) + m[:, :, f] * x[
                        :, :, f
                    ]

        # Initial poitn interpolation for better performance
        imputed_x = initial_point_interpolation(x, m, t, imputed_x, median_vals)

        return imputed_x

    def fc_train(self, x, m, t, median_vals, model_parameters):
        """Train Fully Connected Networks after RNN block.
        
        Args:
            - x: incomplete data
            - m: mask matrix
            - t: time matrix
            - median_vals: median imputation for variables without observations
            - model_parameters:
                - h_dim: hidden state dimensions
                - batch_size: the number of samples in mini-batch
                - iteration: the number of iteration
                - learning_rate: learning rate of model training
        """
        tf.compat.v1.reset_default_graph()

        self.h_dim = model_parameters["h_dim"]
        self.batch_size = model_parameters["batch_size"]
        self.iteration = model_parameters["iteration"]
        self.learning_rate = model_parameters["learning_rate"]

        # rnn imputation results
        rnn_imputed_x = self.rnn_predict(x, m, t, median_vals)

        # Reshape the data for FC train
        x = np.reshape(x, [self.no * self.seq_len, self.dim])
        rnn_imputed_x = np.reshape(rnn_imputed_x, [self.no * self.seq_len, self.dim])
        m = np.reshape(m, [self.no * self.seq_len, self.dim])

        # input place holders
        x_input = tf.placeholder(tf.float32, [None, self.dim])
        target = tf.placeholder(tf.float32, [None, self.dim])
        mask = tf.placeholder(tf.float32, [None, self.dim])

        # build a FC network
        U = tf.compat.v1.get_variable(
            "U", shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer()
        )
        V1 = tf.compat.v1.get_variable(
            "V1", shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer()
        )
        V2 = tf.compat.v1.get_variable(
            "V2", shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer()
        )
        b = tf.Variable(tf.random.normal([self.dim]))

        L1 = tf.nn.sigmoid(
            (
                tf.matmul(x_input, tf.linalg.set_diag(U, np.zeros([self.dim,])))
                + tf.matmul(target, tf.linalg.set_diag(V1, np.zeros([self.dim,])))
                + tf.matmul(mask, V2)
                + b
            )
        )

        W = tf.Variable(tf.random.normal([self.dim]))
        a = tf.Variable(tf.random.normal([self.dim]))
        hypothesis = W * L1 + a

        outputs = tf.nn.sigmoid(hypothesis)

        # reshape out for sequence_loss
        loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - target)))

        # Optimizer
        optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        train = optimizer.minimize(loss)

        # Sessions
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        # Training step
        for i in range(self.iteration * 20):
            batch_idx = np.random.permutation(x.shape[0])[: self.batch_size]
            _, step_loss = sess.run(
                [train, loss],
                feed_dict={x_input: x[batch_idx, :], target: rnn_imputed_x[batch_idx, :], mask: m[batch_idx, :]},
            )

        # Save model
        inputs = {"x_input": x_input, "target": target, "mask": mask}
        outputs = {"imputation": outputs}

        save_file_name = self.save_file_directory + "/fc_feature/"
        tf.compat.v1.saved_model.simple_save(sess, save_file_name, inputs, outputs)

    def rnn_fc_predict(self, x, m, t, median_vals):
        """Impute missing data using RNN and FC.
        
        Args:
            - x: incomplete data
            - m: mask matrix
            - t: time matrix
            - median_vals: median imputation for variables without observations
        
        Returns:
            - fc_imputed_x: imputed data using RNN and FC
        """
        # rnn imputation results
        rnn_imputed_x = self.rnn_predict(x, m, t, median_vals)

        print("Finish M-RNN imputations with RNN")

        # Reshape the data for FC predict
        x = np.reshape(x, [self.no * self.seq_len, self.dim])
        rnn_imputed_x = np.reshape(rnn_imputed_x, [self.no * self.seq_len, self.dim])
        m = np.reshape(m, [self.no * self.seq_len, self.dim])

        save_file_name = self.save_file_directory + "/fc_feature/"

        # Load saved data
        graph = tf.Graph()
        with graph.as_default():
            with tf.compat.v1.Session() as sess:

                sess.run(tf.compat.v1.global_variables_initializer())
                tf.compat.v1.saved_model.loader.load(sess, [tf.compat.v1.saved_model.SERVING], save_file_name)
                x_input = graph.get_tensor_by_name("Placeholder:0")
                target = graph.get_tensor_by_name("Placeholder_1:0")
                mask = graph.get_tensor_by_name("Placeholder_2:0")
                outputs = graph.get_tensor_by_name("Sigmoid_1:0")

                fc_imputed_x = sess.run(outputs, feed_dict={x_input: x, target: rnn_imputed_x, mask: m})

        # Reshape imputed data to 3d array
        fc_imputed_x = np.reshape(fc_imputed_x, [self.no, self.seq_len, self.dim])
        m = np.reshape(m, [self.no, self.seq_len, self.dim])
        x = np.reshape(x, [self.no, self.seq_len, self.dim])

        fc_imputed_x = fc_imputed_x * (1 - m) + x * m
        fc_imputed_x = initial_point_interpolation(x, m, t, fc_imputed_x, median_vals)

        print("Finish M-RNN imputations with RNN and FC")

        return fc_imputed_x

    def fit(self, x, m, t, median_vals, model_parameters):
        """Train the entire MRNN.
        
        Args:
            - x: incomplete data
            - m: mask matrix
            - t: time matrix
            - median_vals: median imputation for variables without observations
            - model_parameters:
                - h_dim: hidden state dimensions
                - batch_size: the number of samples in mini-batch
                - iteration: the number of iteration
                - learning_rate: learning rate of model training
        """
        # Train RNN part
        for f in tqdm(range(self.dim)):
            self.rnn_train(x, m, t, f, model_parameters)
        print("Finish M-RNN training with RNN for imputation")
        # Train FC part
        self.fc_train(x, m, t, median_vals, model_parameters)
        print("Finish M-RNN training with both RNN and FC for imputation")

    def transform(self, x, m, t, median_vals):
        """Impute missing data using the entire MRNN.
        
        Args:
            - x: incomplete data
            - m: mask matrix
            - t: time matrix
            - median_vals: median imputation for variables without observations
            
        Returns:
            - imputed_x: imputed data
        """
        # Impute with both RNN and FC part
        imputed_x = self.rnn_fc_predict(x, m, t, median_vals)

        return imputed_x
