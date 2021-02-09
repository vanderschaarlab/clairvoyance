"""GANITE. Treatment effects model."""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from base import BaseEstimator, PredictorMixin
from utils.data_utils import concate_xs, concate_xt


class GANITE_Model(BaseEstimator, PredictorMixin):
    def __init__(self, hyperparams=None, stack_dim=None, task=None, static_mode=None, time_mode=None):
        """Initialize the GANITE model.

        Args:
            - hyperparams: dictionary with the hyperparameters specifying the architecture of the GANITE model.
            - stack_dim: number of timesteps to stack to obtain input to GANITE model.
            - task: 'classification' or 'regression'
            - static_mode: 'concatenate' or None
            - time_mode: 'concatenate' or None
        """

        super().__init__(task)
        self.hyperparams = hyperparams
        self.stack_dim = stack_dim

        self.static_mode = static_mode
        self.time_mode = time_mode
        self.task = task

    def init_model(self, params):
        """
        Args:
            - params: dictionary of parameters specifying the following dimensions needed for initializing the placeholder
                values of the TensorFlow graph: num_treatments (number of treatments), num_covariates (number of covariates) and
                num_outputs (number of outputs).
        """

        self.num_features = params["num_features"]
        self.num_treatments = params["num_treatments"]
        self.num_outcomes = params["num_outcomes"]

        self.H_Dim1 = self.hidden_dims
        self.H_Dim2 = self.hidden_dims

        self.size_z = self.num_treatments * self.num_outcomes  # Size of Z is equal to num potential outcomes

        tf.reset_default_graph()

        # Feature (X)
        self.X = tf.placeholder(tf.float32, shape=[None, self.num_features], name="input_features")
        # Treatment (T) - one-hot encoding for the treatment
        self.T = tf.placeholder(tf.float32, shape=[None, self.num_treatments], name="input_treatment")
        # Outcome (Y)
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_outcomes], name="input_y")
        # Random Noise (G)
        self.Z_G = tf.placeholder(tf.float32, shape=[None, self.size_z], name="input_noise")
        # Test Outcome (Y_T) - Potential outcome
        self.Y_T = tf.placeholder(tf.float32, shape=[None, 1])

    def generator(self, x, t, y, z):
        """Counterfactual Generator. Generates the counterfactual outcomes.

        Args:
            - x: input features
            - t: factual treatment
            - y: factual outcome
            - z: noise

        Return:
            - G_logit: generated logits for the counterfactual outcomes.
        """

        with tf.variable_scope("generator"):
            inputs = tf.concat(axis=1, values=[x, t, y, z])
            G_shared = tf.layers.dense(inputs, self.H_Dim1, activation=tf.nn.relu, name="shared")
            treatment_logit_predictions = dict()
            for treatment in range(self.num_treatments):
                treatment_layer = tf.layers.dense(
                    G_shared, self.H_Dim2, activation=tf.nn.relu, name="treatment_layer_%s" % str(treatment)
                )
                treatment_output = tf.layers.dense(
                    treatment_layer, 1, activation=None, name="treatment_output_%s" % str(treatment)
                )
                treatment_logit_predictions[treatment] = treatment_output

            G_logit = tf.concat(list(treatment_logit_predictions.values()), axis=-1)

        return G_logit

    def discriminator(self, x, t, y, tilde_y):
        """Counterfactual Discrimnator. Discriminates which outcomes are factual or counterfactual.

        Args:
            - x: input features
            - t: factual treatment
            - y: factual outcome
            - tilde_y: counterfactual_outcomes

        Return:
            - D_logit: logits for all potential outcomes indicating whether they are factual or counterfactual.
        """
        with tf.variable_scope("discriminator"):
            # Factual & Counterfactual outcomes concatenate
            outcomes = t * y + (1.0 - t) * tilde_y
            inputs = tf.concat(axis=1, values=[x, outcomes])

            D_h1 = tf.layers.dense(inputs, self.H_Dim1, activation=tf.nn.relu)
            D_h2 = tf.layers.dense(D_h1, self.H_Dim2, activation=tf.nn.relu)

            D_logit = tf.layers.dense(D_h2, self.num_treatments, activation=None)

        return D_logit

    def inference(self, x):
        """Inference network. Predicts all potential outcomes given input features.

        Args:
         - x: input features

        Returns:
         - I_logit: predicted potential outcomes.
        """

        with tf.variable_scope("inference"):
            inputs = x
            I_shared = tf.layers.dense(inputs, self.H_Dim1, activation=tf.nn.relu, name="shared")

            treatment_logit_predictions = dict()
            for treatment in range(self.num_treatments):
                treatment_layer = tf.layers.dense(
                    I_shared, self.H_Dim2, activation=tf.nn.relu, name="treatment_layer_%s" % str(treatment)
                )
                treatment_output = tf.layers.dense(
                    treatment_layer, 1, activation=None, name="treatment_output_%s" % str(treatment)
                )

                treatment_logit_predictions[treatment] = treatment_output

            I_logit = tf.concat(list(treatment_logit_predictions.values()), axis=-1)

        return I_logit

    def sample_Z(self, m, n):
        """Random sample noize of size [m, n] """
        return np.random.uniform(0, 1.0, size=[m, n])

    def sample_X(self, X, size):
        """Sample a batch of input features."""
        start_idx = np.random.randint(0, X.shape[0], size)
        return start_idx

    def train(self, Train_X, Train_T, Train_Y):
        """Train GANITE model.

        Args:
         - Train_X: training patient features.
         - Train_T: training factual treatments.
         - Train_Y: training factual outcomes.
        """
        # 1. Generator
        G_logits = self.generator(x=self.X, y=self.Y, t=self.T, z=self.Z_G)
        G_prob = tf.nn.sigmoid(G_logits)

        # 2. Discriminator
        D_logits = self.discriminator(x=self.X, t=self.T, y=self.Y, tilde_y=G_logits)

        # 3. Inference
        I_logits = self.inference(self.X)
        I_prob = tf.nn.sigmoid(I_logits)

        G_outcomes = tf.identity(G_logits, name="generator_outcomes")
        I_outcomes = tf.identity(I_logits, name="inference_outcomes")

        # 1. Discriminator loss
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.T, logits=D_logits))

        # 2. Generator loss
        G_loss_GAN = -D_loss
        G_logit_factual = tf.reduce_max(self.T * G_logits, axis=1, keepdims=True)

        if self.task == "regression":
            G_loss_R = tf.sqrt(tf.reduce_mean((self.Y - G_logit_factual) ** 2))
        elif self.task == "classification":
            G_loss_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=G_logit_factual))

        G_loss = self.alpha * (G_loss_R) + G_loss_GAN

        # 3. Inference loss
        if self.task == "regression":
            self.I_factual = tf.reduce_max(self.T * I_logits, axis=1, keepdims=True)
            I_loss1 = tf.sqrt(tf.reduce_mean((G_logits - I_logits) ** 2))
            I_loss2 = tf.sqrt(tf.reduce_mean((self.Y - self.I_factual) ** 2))
            I_loss = I_loss1 + I_loss2

        elif self.task == "classification":
            G_potential_outcomes = self.T * self.Y + (1.0 - self.T) * G_prob
            self.I_factual = tf.reduce_max(self.T * I_prob, axis=1, keepdims=True)
            I_loss1 = tf.sqrt(tf.reduce_mean((G_potential_outcomes - I_prob) ** 2))
            I_loss2 = tf.sqrt(tf.reduce_mean((self.Y - self.I_factual) ** 2))
            I_loss = I_loss1 + I_loss2

        theta_G = tf.trainable_variables(scope="generator")
        theta_D = tf.trainable_variables(scope="discriminator")
        theta_I = tf.trainable_variables(scope="inference")

        # %% Solver
        G_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(G_loss, var_list=theta_G)
        D_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(D_loss, var_list=theta_D)
        I_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(I_loss, var_list=theta_I)

        # Setup tensorflow
        tf_device = "gpu"
        if tf_device == "cpu":
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 0})
        else:
            tf_config = tf.ConfigProto(log_device_placement=False, device_count={"GPU": 1})
            tf_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        # Iterations
        # Train G and D first

        for it in tqdm(range(10000)):

            for kk in range(1):
                idx_mb = self.sample_X(Train_X, self.batch_size)
                X_mb = Train_X[idx_mb]
                T_mb = Train_T[idx_mb]
                Y_mb = Train_Y[idx_mb]
                Z_G_mb = self.sample_Z(self.batch_size, self.size_z)

                _, G_loss_curr, G_logits_curr, G_logit_factual_curr = self.sess.run(
                    [G_solver, G_loss, G_logits, G_logit_factual],
                    feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb, self.Z_G: Z_G_mb},
                )

            for kk in range(1):
                idx_mb = self.sample_X(Train_X, self.batch_size)
                X_mb = Train_X[idx_mb]
                T_mb = Train_T[idx_mb]
                Y_mb = Train_Y[idx_mb]
                Z_G_mb = self.sample_Z(self.batch_size, self.size_z)

                _, D_loss_curr, = self.sess.run(
                    [D_solver, D_loss], feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb, self.Z_G: Z_G_mb}
                )

        # Train I
        for it in tqdm(range(10000), disable=True):
            idx_mb = self.sample_X(Train_X, self.batch_size)
            X_mb = Train_X[idx_mb, :]
            T_mb = Train_T[idx_mb]
            Y_mb = Train_Y[idx_mb]
            Z_G_mb = self.sample_Z(self.batch_size, self.size_z)

            (
                _,
                I_loss_curr,
                I_loss1_curr,
                I_loss2_curr,
                I_logits_curr,
                G_logits_curr,
                I_logit_factual_curr,
            ) = self.sess.run(
                [I_solver, I_loss, I_loss1, I_loss2, I_logits, G_logits, self.I_factual],
                feed_dict={self.X: X_mb, self.T: T_mb, self.Y: Y_mb, self.Z_G: Z_G_mb},
            )

    def data_preprocess(self, dataset, fold, split):
        """Preprocess the dataset.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - split: 'train', 'valid' or 'test'

        Returns:
            -    stacked_dataset: stacked dataset dictionary for training GANITE.
            -    x: original time-series patient features.
        """
        x, s, y, t, treat = dataset.get_fold(fold, split)

        if self.static_mode == "concatenate":
            x = concate_xs(x, s)

        if self.time_mode == "concatenate":
            x = concate_xt(x, t)

        one_hot_treatments = np.zeros(shape=(treat.shape[0], treat.shape[1], 2))
        treat = np.round(treat)

        for patient_id in range(treat.shape[0]):
            for timestep in range(treat.shape[1]):
                if treat[patient_id][timestep][0] == 0.0:
                    one_hot_treatments[patient_id][timestep] = [1, 0]
                elif treat[patient_id][timestep][0] == 1.0:
                    one_hot_treatments[patient_id][timestep] = [0, 1]
                elif treat[patient_id][timestep][0] == -1.0:
                    one_hot_treatments[patient_id][timestep] = [-1, -1]

        active_entries = np.ndarray.max((y >= 0).astype(int), axis=-1)
        sequence_lengths = np.sum(active_entries, axis=1)

        num_features = x.shape[-1]
        num_outcomes = y.shape[-1]
        num_treatments = one_hot_treatments.shape[-1]

        stacked_x_list = []
        stacked_y_list = []
        stacked_treat_list = []
        patient_ids = []

        stack_dim = self.stack_dim
        total = 0
        for (index, patient_trajectory) in enumerate(x):
            trajectory_length = sequence_lengths[index]

            for step in range(trajectory_length):
                total = total + 1
                stacked_x = np.zeros(shape=(stack_dim, num_features))

                patient_ids.append(index)
                stacked_treat_list.append(one_hot_treatments[index][step])
                stacked_y_list.append(y[index][step])
                if step < stack_dim:
                    stacked_x[-step - 1 :] = patient_trajectory[: step + 1]
                else:
                    stacked_x = patient_trajectory[step - stack_dim + 1 : step + 1]
                stacked_x = stacked_x.flatten()
                stacked_x_list.append(stacked_x)

        stacked_dataset = dict()
        stacked_dataset["x"] = np.reshape(np.array(stacked_x_list), newshape=(total, num_features * stack_dim))
        stacked_dataset["y"] = np.reshape(np.array(stacked_y_list), newshape=(total, num_outcomes))
        stacked_dataset["treat"] = np.reshape(np.array(stacked_treat_list), newshape=(total, num_treatments))
        stacked_dataset["patient_ids"] = np.array(patient_ids)
        stacked_dataset["sequence_lengths"] = sequence_lengths

        return stacked_dataset, x

    def fit(self, dataset, fold=0, train_split="train", val_split="val"):
        """Fit the treatment effects model model.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
        """
        stacked_dataset_train, _ = self.data_preprocess(dataset, fold, train_split)
        stacked_dataset_val, _ = self.data_preprocess(dataset, fold, val_split)

        params = {
            "num_features": stacked_dataset_train["x"].shape[-1],
            "num_treatments": stacked_dataset_train["treat"].shape[-1],
            "num_outcomes": stacked_dataset_train["y"].shape[-1],
        }

        self.batch_size = self.hyperparams["batch_size"]
        self.alpha = self.hyperparams["alpha"]
        self.learning_rate = self.hyperparams["learning_rate"]
        self.hidden_dims = self.hyperparams["hidden_dims"]
        self.init_model(params=params)

        # Train the model
        self.train(
            Train_X=stacked_dataset_train["x"],
            Train_Y=stacked_dataset_train["y"],
            Train_T=stacked_dataset_train["treat"],
        )

    def predict(self, dataset, fold=0, test_split="test"):
        """Return the predicted factual outcomes on the test set.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - test_split: testing set splitting parameter

        Returns:
            - test_y_hat: predictions on testing set
        """
        stacked_dataset_test, x = self.data_preprocess(dataset, fold, test_split)

        [predicted_outputs] = self.sess.run(
            [self.I_factual], feed_dict={self.X: stacked_dataset_test["x"], self.T: stacked_dataset_test["treat"]}
        )

        test_y_hat = self.process_sequence_predictions(predicted_outputs, stacked_dataset_test, x)

        return test_y_hat

    def process_sequence_predictions(self, stacked_predictions, stacked_dataset_test, x):
        """Process stacked predictions to create time-series predictions.

        Args:
            - stacked_predictions: stacked predicted outcomes for test patients.
            - x: original time-series patient features.

        Returns:
            - test_y_hat: time series predictions
        """
        sequence_lengths = stacked_dataset_test["sequence_lengths"]
        test_y_hat = np.zeros(shape=(x.shape[0], x.shape[1], stacked_predictions.shape[-1]))

        current_position = 0
        for (index, _) in enumerate(x):
            test_y_hat[index][0 : sequence_lengths[index]] = stacked_predictions[
                current_position : current_position + sequence_lengths[index]
            ]
            current_position = current_position + sequence_lengths[index]

        return test_y_hat
