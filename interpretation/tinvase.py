"""Time-series model interpretation (T-INVASE)

Reference: J. Yoon, J. Jordon, M. van der Schaar, "INVASE: Instance-wise Variable Selection using Neural Networks," ICLR, 2019
"""

# Necessary Packages
from tensorflow.keras.layers import Input, Dense, Masking
from tensorflow.keras.layers import TimeDistributed, Multiply, Lambda, Activation
from tensorflow.keras import Model
import tensorflow as tf
import os
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from utils import concate_xs, concate_xt
from utils import select_loss, rmse_loss, rnn_layer
from base import BaseEstimator, PredictorMixin


class TInvase(BaseEstimator, PredictorMixin):
    """INVASE extension to time-series.
    
    Returns the temporal and feature importance for each sample.
    
    Attributes:
        - predictor_model: predictor_model for interpretation.
        - task: classification or regression
        - model_type: 'rnn', 'lstm', or 'gru'
        - h_dim: hidden dimensions
        - n_head: number of heads (transformer model)
        - n_layer: the number of layers
        - batch_size: the number of samples in each batch
        - epoch: the number of iteration epochs
        - static_mode: 'concatenate' or None
        - time_mode: 'concatenate' or None
        - model_id: the name of model
        - model_path: model path for saving
    """

    def __init__(
        self,
        predictor_model=None,
        task=None,
        model_type=None,
        h_dim=None,
        n_head=None,
        n_layer=None,
        batch_size=None,
        epoch=None,
        learning_rate=None,
        static_mode=None,
        time_mode=None,
        model_id="interpretable_model",
        model_path="tmp",
        verbose=False,
    ):

        super().__init__(predictor_model, task)

        self.predictor_model = predictor_model
        self.task = task

        self.model_type = model_type
        self.h_dim = h_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.static_mode = static_mode
        self.time_mode = time_mode
        self.model_path = model_path
        self.model_id = model_id
        self.verbose = verbose

        # Set path for model saving
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.save_file_name = "{}/{}".format(model_path, model_id) + datetime.now().strftime("%H%M%S") + ".hdf5"

        # Interpretable model define
        self.interpretable_model = None

    def data_preprocess(self, dataset, fold, split):
        """Preprocess the dataset.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - split: 'train', 'valid' or 'test'
            
        Returns:
            - x: temporal feature
            - y_hat: predictions of the predictor model
        """
        # Set temporal, static, and time information
        x, s, _, t, _ = dataset.get_fold(fold, split)
        # Label is the prediction of the predictor
        y_hat = self.predictor_model.predict(dataset, test_split=split)
        # Static & Time information concatenating
        if self.static_mode == "concatenate":
            if s is not None:
                x = concate_xs(x, s)
        if self.time_mode == "concatenate":
            if t is not None:
                x = concate_xt(x, t)

        return x, y_hat

    def _build_model(self, x, y):
        """Construct the model using feature and label statistics.
        
        Args:
            - x: temporal feature
            - y: labels
            
        Returns:
            - model: interpretation model
        """
        # Parameters
        dim = len(x[0, 0, :])
        max_seq_len = len(x[0, :, 0])

        # Build one input, two outputs model
        main_input = Input(shape=(max_seq_len, dim), dtype="float32")
        mask_layer = Masking(mask_value=-1.0)(main_input)

        select_layer = rnn_layer(mask_layer, self.model_type, self.h_dim, return_seq=True)

        for _ in range(self.n_layer):
            select_layer = rnn_layer(select_layer, self.model_type, self.h_dim, return_seq=True)

        select_layer = TimeDistributed(Dense(dim, activation="sigmoid"))(select_layer)

        select_layer = Lambda(lambda x: x - 0.5)(select_layer)
        select_layer = Activation("relu")(select_layer)
        select_out = Lambda(lambda x: x * 2, name="select")(select_layer)

        # Second output
        pred_layer = Multiply()([mask_layer, select_out])

        for _ in range(self.n_layer - 1):
            pred_layer = rnn_layer(pred_layer, self.model_type, self.h_dim, return_seq=True)

        return_seq_bool = len(y.shape) == 3
        pred_layer = rnn_layer(pred_layer, self.model_type, self.h_dim, return_seq_bool)

        if self.task == "classification":
            act_fn = "sigmoid"
        elif self.task == "regression":
            act_fn = "linear"

        if len(y.shape) == 3:
            pred_out = TimeDistributed(Dense(y.shape[-1], activation=act_fn), name="predict")(pred_layer)
        elif len(y.shape) == 2:
            pred_out = Dense(y.shape[-1], activation=act_fn, name="predict")(pred_layer)

        model = Model(inputs=main_input, outputs=[select_out, pred_out])

        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

        model.compile(
            loss={"select": select_loss, "predict": rmse_loss},
            optimizer=adam,
            loss_weights={"select": 0.01, "predict": 1},
        )

        return model

    def fit(self, dataset, fold=0, train_split="train", valid_split="val"):
        """Fit the interpretation model.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter
            
        Returns:
            - self.interpretor_model: trained interpretor model
        """
        # Predictor model can be any model. But TINVASE is based on RNN
        if self.model_type not in ["rnn", "lstm", "gru"]:
            self.model_type = "gru"
        elif self.model_type is None:
            self.model_type = "gru"

        # Train / Valid datasets
        train_x, train_y = self.data_preprocess(dataset, fold, train_split)
        valid_x, valid_y = self.data_preprocess(dataset, fold, valid_split)

        # Build model
        self.interpretor_model = self._build_model(train_x, train_y)

        # Callback for the best model saving
        save_best = ModelCheckpoint(
            self.save_file_name, monitor="val_loss", mode="min", verbose=False, save_best_only=True
        )

        # Train the model
        self.interpretor_model.fit(
            train_x,
            [train_x, train_y],
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(valid_x, [valid_x, valid_y]),
            callbacks=[save_best],
            verbose=self.verbose,
        )

        self.interpretor_model.load_weights(self.save_file_name)
        os.remove(self.save_file_name)

        return self.interpretor_model

    def predict(self, dataset, fold=0, test_split="test"):
        """Return the temporal and feature importance.
        
        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - test_split: testing set splitting parameter
            
        Returns:
            - test_s_hat: temporal and feature importance in 3d array    
        """
        test_x, _ = self.data_preprocess(dataset, fold, test_split)
        test_s_hat, _ = self.interpretor_model.predict(test_x)
        return test_s_hat

    def save_model(self, model_path):
        """Save the model to model_path
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        self.interpretor_model.save(model_path)

    def load_model(self, model_path):
        """Load and return the model from model_path        
        
        Args:
            - model_path: path of the saved model (it should be .h5)
        """
        assert model_path[-3:] == ".h5"
        assert os.path.exists(model_path) is True

        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        loaded_model.compile(loss={"select": select_loss, "predict": rmse_loss}, optimizer=self.adam)

        self.interpretor_model = loaded_model
        return loaded_model
