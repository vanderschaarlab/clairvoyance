"""AutoML for sequence prediction

Reference:    Y. Zhang, D. Jarrett, M. van der Schaar, "Stepwise Model Selection for Sequence Prediction
via Deep Kernel Learning," International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.
"""

import tensorflow as tf
import numpy as np
from scipy.special import erfc
from tensorflow.python.ops import rnn

from automl._utils import init_random_uniform, error_function, model_eval, get_opt_domain


def _get_quantiles(acquisition_par, fmin, m, s):
    if isinstance(s, np.ndarray):
        s[s < 1e-10] = 1e-10
    elif s < 1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par) / s
    phi = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return phi, Phi, u


def _compute_acq(m, s, fmin, jitter):
    phi, Phi, u = _get_quantiles(jitter, fmin, m, s)
    f_acqu = s * (u * Phi + phi)
    return f_acqu


def _L_cholesky(x, DD):
    jitter = 1e-15

    L_matrix = tf.linalg.cholesky(x + jitter * tf.eye(DD, dtype=tf.float32))
    return L_matrix


class AutoTS:
    """Automated hyperparameter tuning for time series prediction.

    References:
            Y. Zhang, D. Jarrett, M. van der Schaar, "Stepwise Model Selection for Sequence Prediction
            via Deep Kernel Learning," International Conference on Artificial Intelligence and Statistics (AISTATS), 2020.

    """

    def __init__(
        self, dataset, model, metric, rnn_hidden_size=64, h_size=32, num_h=32, rho_size=2, lr=0.01, model_path=None
    ):
        """
        Args:
            dataset: A :class:`~datasets.PandasDataset` object with *train and validation split defined*.
            model: A predictor, for example, :class:`~prediction.GeneralRNN`.
            metric: A :class:`~evaluation.BOMetric` object, which measures model performance.
            rnn_hidden_size: (Advanced) RNN hidden size of the AutoML model.
            h_size: (Advanced) Hidden size of the AutoML model.
            num_h: (Advanced) Number of Hidden of the AutoML model.
            rho_size: (Advanced) Rho size of the AutoML model.
            lr: (Advanced) Learning rate of the AutoML model.

        Notes:
            The parameters marked 'Advanced' is for developers only.
            End users are recommended to accepted the default.

        Attributes:
            - All: All attributes of this class are private.

        """

        data_x, bounds, BO_data, lim_domain, candidate_model = AutoTS._init_eval(dataset, model, metric)

        self.model_path = model_path
        self.dataset = dataset
        self.metric = metric
        self.init_model = candidate_model
        self.trainX, self.trainY = BO_data
        self.dims = [np.shape(self.trainX)[1], num_h, num_h, num_h, num_h]
        self.bounds, self.bound_type = bounds[0], bounds[1]
        self.rnn_hidden_size = rnn_hidden_size
        self.num_example = np.shape(data_x)[0]
        self.max_length = np.shape(data_x)[1]
        self.lim_domain = lim_domain
        self.rho_size = rho_size
        self.data_X = data_x
        self.params = dict()
        self.n_layers = len(self.dims) - 1
        self.lr = lr
        self.feature_vector = {}
        self.ml_primal = {}
        self.ker_inv = {}
        self.mean = {}
        self.weight_deepset = {}
        self.bias_deepset = {}
        self.weight_bo = {}
        self.bias_bo = {}
        self.embed = {}
        self.beta = {}
        self.lam = {}
        self.r = {}
        self.Y = {}
        self.X = {}

        hidden_size = [
            [self.rnn_hidden_size, h_size],
            [h_size, h_size],
            [h_size, h_size],
            [h_size, h_size],
            [h_size, rho_size],
        ]

        # create variables
        for i in range(len(hidden_size)):
            self._create_variable("deepset/layer%d" % i, "weight", hidden_size[i])
            self._create_variable("deepset/layer%d" % i, "biases", hidden_size[i][-1])

            self.weight_deepset[str(i)] = self._get_variable("deepset/layer%d" % i, "weight", True)
            self.bias_deepset[str(i)] = self._get_variable("deepset/layer%d" % i, "biases", True)

        for i in range(self.n_layers):
            self._create_variable("bo_net/layer%d" % i, "weight", [self.dims[i], self.dims[i + 1]])
            self._create_variable("bo_net/layer%d" % i, "biases", [self.dims[i + 1]])
            self.weight_bo[str(i)] = self._get_variable("bo_net/layer%d" % i, "weight", True)
            self.bias_bo[str(i)] = self._get_variable("bo_net/layer%d" % i, "biases", True)

        for i in range(self.max_length):
            self._create_variable("bo_net/task%d" % i, "lam", [1, 1])
            self._create_variable("bo_net/task%d" % i, "beta", [1, 1])

            lam = self._get_variable("bo_net/task%d" % i, "lam", True)
            self.lam[str(i)] = tf.math.softplus(lam)

            beta = self._get_variable("bo_net/task%d" % i, "beta", True)
            self.beta[str(i)] = tf.math.softplus(beta)

            self.r[str(i)] = self.beta[str(i)] / self.lam[str(i)]

        self._build_model()

        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        for i in range(self.max_length):
            with tf.compat.v1.variable_scope("bo_net/task%d" % i, reuse=True):
                var6 = tf.compat.v1.get_variable("lam")
                var7 = tf.compat.v1.get_variable("beta")

            var6 = tf.compat.v1.assign(var6, 100 * tf.ones(tf.shape(var6)))
            var7 = tf.compat.v1.assign(var7, 100 * tf.ones(tf.shape(var7)))

            self.sess.run(var6)
            self.sess.run(var7)

    def training_loop(self, num_iter=5):
        """
        Run AutoML algorithm with initialized values.

        Args:
            num_iter: Number of Bayesian Optimization iterations.

        Returns:
            - List of models trained with different hyperparameters during the optimization.
            - The performance of the models. Use this value to create a model ensemble with :class:`~prediction.AutoEnsemble`.
        """
        dataset = self.dataset
        metric = self.metric
        model_list = [self.init_model]
        rmse_global = []
        rmse_wise = []
        domain_keys = [x["name"] for x in self.lim_domain]
        assert num_iter > 0
        BO_output = 0

        for i in range(num_iter):

            # Optimize the BO network
            if i % 5 == 0:
                try:
                    self._batch_optimization()
                except Exception:
                    print("Covariance matrix not invertible.")

            # Make a acquistion
            query = self._acquisition()
            param_dict = dict(zip(domain_keys, query))

            # Evaluate the acquistion
            id_parts = self.init_model.model_id.split("_")
            id_parts[-1] = str(i)
            new_model_id = "_".join(id_parts)

            model_candidate = self.init_model.new(new_model_id)
            model_candidate.set_params(**param_dict)

            obs = model_eval(dataset, model_candidate, metric)

            model_list.append(model_candidate)
            # Update the acquisition set
            BO_input, BO_output = self._update_data(query, obs)

            print("Number of Function Evaluations:", i)
            print("query point:", query)

            # Compute the minimum rmse obtained by global and step-wise model selection
            rmse1, rmse2 = error_function(BO_output)

            # Compare the convergence plot of global and step-wise model selection
            print("minimum loss of global model selection:", rmse1)
            print("minimum loss of step-wise model selection:", rmse2)

            rmse_global.append(rmse1)
            rmse_wise.append(rmse2)

        if self.model_path is not None:
            for m in model_list:
                m.save_model(self.model_path + "/" + m.model_id + ".h5")
        self._destroy_graph()
        return model_list, BO_output

    @staticmethod
    def _init_eval(dataset, model, metric):
        """
        Provides initial values for AutoML algorithm.

        Args:
            dataset: A :class:`~datasets.PandasDataSet` object with train and validation split defined.
            model: A predictor, for example, :class:`~prediction.GeneralRNN`.
            metric: A :class:`~evaluation.BOMetric` object, which measures model performance.

        Returns:
            - Data matrix used to initialize :class:`~automl.AutoTS`.
            - Parameter bounds used to initialize :class:`~automl.AutoTS`.
            - Initial values used to initialize :class:`~automl.AutoTS`.
            - Parameter domains used to initialize :class:`~automl.AutoTS`.
            - The initial model.

        """
        # get domain
        model_domain = model.get_hyperparameter_space()
        lim_domain, dim, bounds = get_opt_domain(model_domain)
        domain_keys = [x["name"] for x in model.get_hyperparameter_space()]

        # initial run
        list_domain = init_random_uniform(lim_domain, n_points=1, initial=True)[0]
        param_dict = dict(zip(domain_keys, list_domain))

        candidate_model = model.new(model.model_id + "bo_iter_0")
        candidate_model.set_params(**param_dict)

        obs = model_eval(dataset, candidate_model, metric)
        BO_data = np.array(list_domain)[None, :], obs

        # set input matrix
        seq_len = dataset.temporal_feature.shape[1]
        if dataset.problem == "one-shot":
            label = np.tile(dataset.label[:, None, :], (1, seq_len, 1))
        else:
            label = dataset.label

        if dataset.static_feature is not None:
            sf = np.tile(dataset.static_feature[:, None, :], (1, seq_len, 1))
            data_x = np.concatenate((dataset.temporal_feature, sf, label), axis=-1)
        else:
            data_x = np.concatenate((dataset.temporal_feature, label), axis=-1)
        return data_x, bounds, BO_data, lim_domain, candidate_model

    def _build_model(self):
        self.X = tf.compat.v1.placeholder(tf.float32, [None, self.dims[0]])
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, self.max_length])
        self.N_sample = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1])
        lstm_cell = tf.keras.layers.LSTMCell(units=self.rnn_hidden_size)

        self.segment_ids = tf.compat.v1.placeholder(tf.float32, [self.max_length])
        self.rnn_dim_feature = np.shape(self.data_X)[-1]
        self.rnn_input = tf.constant(
            value=self.data_X, dtype=tf.float32, shape=[self.num_example, self.max_length, self.rnn_dim_feature]
        )

        state = lstm_cell.get_initial_state(batch_size=self.num_example, dtype=tf.float32)
        xx = tf.unstack(self.rnn_input, self.max_length, 1)
        rnn_output, state = rnn.static_rnn(lstm_cell, xx, initial_state=state)
        #        rnn_output, state = tf.keras.layers.SimpleRNN(lstm_cell, return_state = True)(xx, initial_state=state)

        self.train_loss = 0
        self.YY = {}

        for step in range(self.max_length):
            self.YY[str(step)] = tf.slice(self.Y, [0, step], [-1, 1])

            ss = tf.slice(rnn_output, [step, 0, 0], [1, -1, -1])

            self.embed[str(step)] = self._deepset(tf.squeeze(ss))

            state_embedding = tf.tile(self.embed[str(step)], [tf.cast(self.N_sample[0], tf.int32), 1])

            self._bo_net(self.X, state_embedding, step)

            self.train_loss += self.ml_primal[str(step)]

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.train_loss)

    def _batch_optimization(self, max_iter=500):

        num_step = min(10, int(self.max_length))
        for i in range(max_iter):
            rr = np.random.randint(0, self.max_length, num_step)
            ss = np.zeros(self.max_length)
            ss[rr] = 1

            feed_input = {
                self.Y: self.trainY,
                self.X: self.trainX,
                self.N_sample: [self.trainX.shape[0]],
                self.segment_ids: ss,
            }

            self.sess.run(self.opt, feed_dict=feed_input)

    def _update_data(self, query, obs):

        query = np.expand_dims(query, axis=1)
        query = np.swapaxes(query, 0, 1)
        self.trainX = np.concatenate([self.trainX, query], axis=0)

        self.trainY = np.concatenate([self.trainY, obs], axis=0)

        return self.trainX, self.trainY

    def _destroy_graph(self):
        tf.reset_default_graph()

    def _get_params(self):
        mdict = dict()
        for scope_name, param in self.params.items():
            w = self.sess.run(param)
            mdict[scope_name] = w
        return mdict

    def _create_variable(self, scope, name, shape, trainable=True):

        with tf.compat.v1.variable_scope(scope):
            w = tf.compat.v1.get_variable(name, shape, trainable=trainable)
        self.params[w.name] = w

    def _get_variable(self, scope, name, trainable=True):
        with tf.compat.v1.variable_scope(scope, reuse=True):
            w = tf.compat.v1.get_variable(name, trainable=trainable)
        return w

    def _deepset(self, hidden_vec):

        Z0 = tf.add(tf.matmul(hidden_vec, self.weight_deepset["0"]), self.bias_deepset["0"])
        A0 = tf.nn.relu(Z0)

        Z1 = tf.add(tf.matmul(A0, self.weight_deepset["1"]), self.bias_deepset["1"])
        A1 = Z1

        mean_emb = tf.reduce_mean(A1, axis=0)
        mean_emb = tf.expand_dims(mean_emb, axis=0)

        Z2 = tf.add(tf.matmul(mean_emb, self.weight_deepset["2"]), self.bias_deepset["2"])
        A2 = tf.nn.relu(Z2)

        Z3 = tf.add(tf.matmul(A2, self.weight_deepset["3"]), self.bias_deepset["3"])
        A3 = tf.nn.relu(Z3)

        Z4 = tf.add(tf.matmul(A3, self.weight_deepset["4"]), self.bias_deepset["4"])

        return Z4

    def _bo_net(self, feature_vector, state_embedding, step):

        for i in range(self.n_layers):
            feature_vector = tf.nn.tanh(tf.matmul(feature_vector, self.weight_bo[str(i)]) + self.bias_bo[str(i)])

        self.feature_vector[str(step)] = tf.concat([state_embedding, feature_vector], axis=1)

        self.DD = tf.shape(self.feature_vector[str(step)])[1]

        phi_phi = self.r[str(step)] * tf.matmul(
            tf.transpose(self.feature_vector[str(step)]), self.feature_vector[str(step)]
        )

        Ker = phi_phi + tf.eye(self.DD, dtype=tf.float32)

        L_matrix = _L_cholesky(Ker, self.DD)

        L_inv_reduce = tf.linalg.triangular_solve(L_matrix, rhs=tf.eye(self.DD, dtype=tf.float32))

        L_y = tf.matmul(L_inv_reduce, tf.matmul(tf.transpose(self.feature_vector[str(step)]), self.YY[str(step)]))

        term1 = (
            0.5
            * self.beta[str(step)]
            * (tf.reduce_sum(tf.square(self.YY[str(step)])) - self.r[str(step)] * tf.reduce_sum(tf.square(L_y)))
        )

        term2 = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_matrix)))

        term3 = -0.5 * self.N_sample[0] * tf.math.log(self.beta[str(step)])

        self.ml_primal[str(step)] = term1 + term2 + term3

        self.ker_inv[str(step)] = tf.matmul(tf.transpose(L_inv_reduce), L_inv_reduce)

        self.mean[str(step)] = self.r[str(step)] * tf.matmul(tf.transpose(L_inv_reduce), L_y)

    def _acquisition(self):

        domain_l = init_random_uniform(self.lim_domain)

        mean_list, XX_inv_list, featureX_list = self.sess.run(
            [self.mean, self.ker_inv, self.feature_vector],
            feed_dict={
                self.X: self.trainX,
                self.Y: self.trainY,
                self.N_sample: [self.trainX.shape[0]],
                self.segment_ids: np.ones(self.max_length),
            },
        )

        test_X_list = self.sess.run(
            self.feature_vector,
            feed_dict={
                self.X: domain_l,
                self.N_sample: [np.shape(domain_l)[0]],
                self.segment_ids: np.ones(self.max_length),
            },
        )

        max_ei = np.zeros(self.max_length)
        index_list = []
        ei_total = []

        for i in range(self.max_length):

            lam = self.sess.run(self.lam[str(i)])

            mean, XX_inv, _, test_X = mean_list[str(i)], XX_inv_list[str(i)], featureX_list[str(i)], test_X_list[str(i)]

            s = []
            for row in range(test_X.shape[0]):
                x = test_X[[row], :]
                s_save = (1 / lam * np.dot(np.dot(x, XX_inv), x.T)) ** 0.5
                s.append(s_save)

            prediction = np.dot(test_X, mean)

            sig = np.reshape(np.asarray(s), (test_X.shape[0], 1))

            ei = _compute_acq(prediction, sig, np.min(self.trainY[:, i]), 0.01)

            ei = np.squeeze(ei)

            ei_total.append(ei)

            anchor_index = ei.argsort()[-1]

            max_ei[i] = np.max(ei)

            index_list.append(anchor_index)

        prob = np.log(max_ei + 1e-15) - np.log(np.sum(max_ei + 1e-15))

        prob = np.exp(prob)

        prob = np.ones_like(prob) / self.max_length

        final_point = domain_l[index_list[np.where(np.random.multinomial(1, prob) > 0.1)[0][0]]]

        return final_point
