"""Transformer modules for time-series prediction.

Reference: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.
"""

# Necessary packages
import os
from datetime import datetime
import torch.nn as nn
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from base import BaseEstimator, PredictorMixin


class TransformerModule(nn.Module):
    """Base transformer module.
    
    Attributes:
        - task: classification or regression
        - problem: one-shot or online
        - dim_input: input dimensions
        - dim_output: output dimensions
        - n_head: number of attention heads
        - n_layer: the number of layers
    """

    def __init__(self, task, problem, dim_input, dim_model, dim_output, num_heads=2, num_layers=2):
        super().__init__()
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.dim_output = dim_output
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.task = task
        self.problem = problem

        self.projection_layer = nn.Linear(dim_input, dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.projection_layer_out = nn.Sequential(
            nn.Linear(dim_model, dim_model + 1), nn.ReLU(inplace=True), nn.Linear(dim_model + 1, dim_output)
        )

        self.src_mask = None

    def forward(self, x):
        # x: (S, N, D)
        x = x.permute((1, 0, 2))
        seq_len = x.shape[0]
        # (S, N)
        seq_mask = x[:, :, 0] != -1
        # (N, S)
        src_key_padding_mask = (x[:, :, 0] == -1).permute((1, 0))

        if self.problem == "online" and self.src_mask is None:
            src_mask = torch.ones((seq_len, seq_len)) * float("-inf")
            self.src_mask = torch.triu(src_mask, diagonal=1).to(x)

        emb = self.projection_layer(x)
        enc = self.transformer(emb, src_key_padding_mask=src_key_padding_mask, mask=self.src_mask)

        if self.problem == "one-shot":
            # aggregate by averaging
            seq_mask = seq_mask * 1.0
            enc = torch.sum(enc * seq_mask[:, :, None], dim=0) / torch.sum(seq_mask[:, :, None], dim=0)

        # S, N, D (oneline)     N, D (one-shot)
        out = self.projection_layer_out(enc)

        # N, D
        if self.problem == "online":
            out = out.permute((1, 0, 2))

        if self.task == "classification":
            out = torch.nn.functional.sigmoid(out)
        return out

    def predict(self, x):
        return self.forward(x)

    def loss_fn(self, y_hat, y):
        if self.task == "classification":
            loss = torch.sum(torch.log(y_hat + 1e-9) * y + torch.log(1.0 - y_hat + 1e-9) * (1.0 - y)) * -1.0
        else:
            loss = torch.sqrt(torch.mean((y_hat - y) ** 2))
        return loss


class TransformerPredictor(BaseEstimator, PredictorMixin):
    """Transformer model for for time-series prediction.

    Ref: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.

    Attributes:
        - task: classification or regression
        - h_dim: hidden dimensions
        - n_layer: the number of layers
        - n_head: number of attention heads
        - batch_size: the number of samples in each batch
        - epoch: the number of iteration epochs
        - learning_rate: the learning rate for the optimizer
        - static_mode: 'concatenate' or None
        - time_mode: 'concatenate' or None
        - model_id: the name of model
        - model_path: model path for checkpointing
        - verbose: print intermediate process
    """

    def __init__(
        self,
        task=None,
        h_dim=None,
        n_layer=None,
        n_head=None,
        batch_size=None,
        epoch=None,
        learning_rate=None,
        static_mode=None,
        time_mode=None,
        model_id="transformer_model",
        model_path="tmp",
        device=None,  # if you have multi GPUs and want to choose which one to use
        verbose=False,
    ):
        super().__init__(task)

        self.task = task
        self.h_dim = h_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.static_mode = static_mode
        self.time_mode = time_mode
        self.model_path = model_path
        self.model_id = model_id
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.predictor_model = None
        self.optimizer = None

        # Set path for model saving
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.save_file_name = "{}/{}".format(model_path, model_id) + datetime.now().strftime("%H%M%S") + ".pth"

    @staticmethod
    def get_hyperparameter_space():
        hyp_ = [
            {"name": "h_dim", "type": "discrete", "domain": [32, 64, 128, 256, 512], "dimensionality": 1},
            {"name": "n_layer", "type": "discrete", "domain": [1, 2, 3, 4], "dimensionality": 1},
            {"name": "n_head", "type": "discrete", "domain": [2, 4, 8], "dimensionality": 1},
            {"name": "batch_size", "type": "discrete", "domain": list(range(100, 1001, 100)), "dimensionality": 1},
            {"name": "learning_rate", "type": "continuous", "domain": [0.0005, 0.005], "dimensionality": 1},
        ]

        return hyp_

    def new(self, model_id):
        """Create a new model with the same parameter as the existing one.

        Args:
            - model_id: an unique identifier for the new model

        Returns:
            - a new GeneralRNN
        """
        return TransformerPredictor(
            self.task,
            self.h_dim,
            self.n_layer,
            self.n_head,
            self.batch_size,
            self.epoch,
            self.learning_rate,
            self.static_mode,
            self.time_mode,
            model_id,
            self.model_path,
            self.device,
            self.verbose,
        )

    def _make_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def fit(self, dataset, fold=0, train_split="train", valid_split="val"):
        """Fit the predictor model.

        Args:
            - dataset: temporal, static, label, time, treatment information
            - fold: Cross validation fold
            - train_split: training set splitting parameter
            - valid_split: validation set splitting parameter

        Returns:
            - self.predictor_model: trained predictor model
        """
        train_x, train_y = self._data_preprocess(dataset, fold, train_split)
        valid_x, valid_y = self._data_preprocess(dataset, fold, valid_split)

        train_dataset = torch.utils.data.dataset.TensorDataset(self._make_tensor(train_x), self._make_tensor(train_y))
        valid_dataset = torch.utils.data.dataset.TensorDataset(self._make_tensor(valid_x), self._make_tensor(valid_y))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

        if self.predictor_model is None:
            self.predictor_model = TransformerModule(
                self.task, dataset.problem, train_x.shape[-1], self.h_dim, train_y.shape[-1], self.n_head, self.n_layer
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.predictor_model.parameters(), lr=self.learning_rate)

        self.predictor_model.train()

        # classification vs regression
        # static vs dynamic
        trainer = create_supervised_trainer(self.predictor_model, self.optimizer, self.predictor_model.loss_fn)
        evaluator = create_supervised_evaluator(
            self.predictor_model, metrics={"loss": Loss(self.predictor_model.loss_fn)}
        )
        # model check point
        checkpoint_handler = ModelCheckpoint(
            self.model_path, self.model_id, n_saved=1, create_dir=True, require_empty=False
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpoint_handler, {"model": self.predictor_model})

        # early stopping
        def score_function(engine):
            val_loss = engine.state.metrics["loss"]
            return -val_loss

        early_stopping_handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)

        # evaluation loss
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            print("Validation Results - Epoch[{}] Avg loss: {:.2f}".format(trainer.state.epoch, metrics["loss"]))

        trainer.run(train_loader, max_epochs=self.epoch)

        return self.predictor_model

    def predict(self, dataset, fold=0, test_split="test"):
        """Predict on the new dataset by the trained model.

                Args:
                    - dataset: temporal, static, label, time, treatment information
                    - fold: Cross validation fold
                    - test_split: testing set splitting parameter

                Returns:
                    - test_y_hat: predictions on the new dataset
        """
        with torch.no_grad():
            self.predictor_model.eval()
            test_x, _ = self._data_preprocess(dataset, fold, test_split)
            test_x = self._make_tensor(test_x)
            test_y_hat = self.predictor_model.predict(test_x).cpu().numpy()
        return test_y_hat

    def save_model(self, model_path):
        torch.save(self.predictor_model, model_path)

    def load_model(self, model_path):
        self.predictor_model = torch.load(model_path)
        self.predictor_model.eval()
