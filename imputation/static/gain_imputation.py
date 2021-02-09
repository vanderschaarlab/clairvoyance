"""GAIN Imputation for static data.

Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
"""

# Necessary packages
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import shutil
from tensorflow.saved_model import tag_constants
from imputation.imputation_utils import rounding
from base import BaseEstimator, DataPreprocessorMixin


class GainImputation(BaseEstimator, DataPreprocessorMixin):
    """GAIN imputation module.
    
    Attributes:
        - file_name: file_name for saving the trained model
    """

    def __init__(self, file_name):
        # File name for saving the trained model
        assert file_name is not None
        self.save_file_directory = "tmp/" + file_name + "/"
        # Normalization parameters
        self.norm_parameters = None

    def sample_Z(self, m, n):
        """Random sample generator for Z.
        
        Args:
            - m: number of rows
            - n: number of columns
            
        Returns:
            - z: generated random values
        """
        z = np.random.uniform(0.0, 0.01, size=[m, n])
        return z

    def fit(self, x):
        """Fit GAIN model.
        
        Args:
            - x: incomplete dataset
        """
        tf.reset_default_graph()

        ## System Parameters
        # Mini batch size
        mb_size = 128
        # Iterations
        iterations = 10000
        # Hint rate
        p_hint = 0.9
        # Loss hyperparameters
        alpha = 10

        # Parameters
        no = len(x)
        dim = len(x[0, :])

        # Hidden state dimensions
        h_dim = dim

        # MinMaxScaler normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        for i in range(dim):
            min_val[i] = np.nanmin(x[:, i])
            x[:, i] = x[:, i] - np.nanmin(x[:, i])
            max_val[i] = np.nanmax(x[:, i])
            x[:, i] = x[:, i] / (np.nanmax(x[:, i]) + 1e-8)

        # Set missing
        m = 1 - (1 * (np.isnan(x)))
        x = np.nan_to_num(x)

        ## Necessary Functions
        # Xavier Initialization Definition
        def xavier_init(size):
            in_dim = size[0]
            xavier_stddev = 1.0 / tf.sqrt(in_dim / 2.0)
            return tf.random_normal(shape=size, stddev=xavier_stddev)

        # Hint Vector Generation
        def sample_M(m, n, p):
            unif_prob = np.random.uniform(0.0, 1.0, size=[m, n])
            M = unif_prob > p
            M = 1.0 * M
            return M

        # Mini-batch generation
        def sample_idx(m, n):
            idx = np.random.permutation(m)
            idx = idx[:n]
            return idx

        ## GAIN Architecture
        # Input Placeholders
        # Mask Vector
        M = tf.compat.v1.placeholder(tf.float32, shape=[None, dim])
        # Hint vector
        H = tf.placeholder(tf.float32, shape=[None, dim])
        # Data with missing values
        X = tf.placeholder(tf.float32, shape=[None, dim])

        # 2. Discriminator
        D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
        D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

        D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

        D_W3 = tf.Variable(xavier_init([h_dim, dim]))
        D_b3 = tf.Variable(tf.zeros(shape=[dim]))

        theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

        # 3. Generator
        G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
        G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

        G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
        G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

        G_W3 = tf.Variable(xavier_init([h_dim, dim]))
        G_b3 = tf.Variable(tf.zeros(shape=[dim]))

        theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

        ## GAIN Function
        # 1. Generator
        def generator(x, m):
            inputs = tf.concat(axis=1, values=[x, m])
            G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
            G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
            G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
            return G_prob

        # 2. Discriminator
        def discriminator(x, h):
            inputs = tf.concat(axis=1, values=[x, h])
            D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
            D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
            D_logit = tf.matmul(D_h2, D_W3) + D_b3
            D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output
            return D_prob

        ## Structure
        # Generator
        G_sample = generator(X, M)

        # Combine with original data
        X_hat = X * M + G_sample * (1 - M)

        # Discriminator
        D_prob = discriminator(X_hat, H)

        ## Loss
        D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1.0 - D_prob + 1e-8))
        G_loss1 = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
        MSE_train_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

        D_loss = D_loss1
        G_loss = G_loss1 + alpha * MSE_train_loss

        ## Solver
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

        # Sessions
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        ## Start Iterations
        for it in tqdm(range(iterations)):

            # Inputs
            mb_idx = sample_idx(no, mb_size)
            x_mb = x[mb_idx, :]
            m_mb = m[mb_idx, :]

            z_mb = self.sample_Z(mb_size, dim)
            h_mb = sample_M(mb_size, dim, 1 - p_hint)
            h_mb = m_mb * h_mb

            x_mb = m_mb * x_mb + (1 - m_mb) * z_mb
            # Loss
            _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict={M: m_mb, X: x_mb, H: h_mb})
            _, G_loss_curr = sess.run([G_solver, G_loss1], feed_dict={M: m_mb, X: x_mb, H: h_mb})

        # Reset the directory for saving
        if not os.path.exists(self.save_file_directory):
            os.makedirs(self.save_file_directory)
        else:
            shutil.rmtree(self.save_file_directory)

        # Save model
        inputs = {"X": X, "M": M}
        outputs = {"imputation": G_sample}
        tf.saved_model.simple_save(sess, self.save_file_directory, inputs, outputs)

        # Parameters for rescaling
        self.norm_parameters = {"min": min_val, "max": max_val}

    def transform(self, data):
        """Return imputed data by trained GAIN model.
        
        Args:
            - data: 2d numpy array with missing data
            
        Returns:
            - imputed data: 2d numpy array without missing data
        """
        # Only after fitting
        assert self.norm_parameters is not None
        assert os.path.exists(self.save_file_directory) is True

        min_val = self.norm_parameters["min"]
        max_val = self.norm_parameters["max"]

        no, dim = data.shape
        # Set missing
        m = 1 - (1 * (np.isnan(data)))
        x = np.nan_to_num(data)

        ## Imputed data
        z = self.sample_Z(no, dim)
        x = m * x + (1 - m) * z

        # Restore the trained GAIN model
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                tf.saved_model.loader.load(sess, [tag_constants.SERVING], self.save_file_directory)
                X = graph.get_tensor_by_name("Placeholder_2:0")
                M = graph.get_tensor_by_name("Placeholder:0")
                G_sample = graph.get_tensor_by_name("Sigmoid:0")

                imputed_data = sess.run(G_sample, feed_dict={X: x, M: m})

        # Renormalize
        for i in range(dim):
            imputed_data[:, i] = imputed_data[:, i] * (max_val[i] + 1e-8)
            imputed_data[:, i] = imputed_data[:, i] + min_val[i]

        # Rounding
        imputed_data = rounding(x, imputed_data)

        return imputed_data
