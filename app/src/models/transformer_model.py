import logging
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger()
torch.classes.__path__ = []


class TransformerModel:
    """
    Wrapper for a PyTorch Transformer model
    """

    def __init__(
        self,
        input_window=10,
        output_window=1,
        batch_size=250,
        lr=0.00005,
        num_epochs=150,
        verbose=0,
    ) -> None:
        self.input_window = input_window
        self.output_window = output_window
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i : i + tw]
            train_label = input_data[
                i + self.output_window : i + tw + self.output_window
            ]
            inout_seq.append((train_seq, train_label))
        return torch.FloatTensor(inout_seq)

    def convert_input(self, X):
        # Transformer model predict next steps of the same series
        # As our target (y_train) is different (it's "buy" or "sell" signal), we won't use it
        # Instead we're going to sum all features and use it as a target series
        # (NOTE: it is a deeply wrong approach as features have different meaning, but I don't see how I can get anything from 2d ndarray)

        # Series can have negative values, avoid them
        series = X.sum(axis=1)
        series = series - min(series) + 0.00001  # eps
        series = np.diff(np.log(series))
        series = series.cumsum()
        series = 2 * series  # Training data augmentation

        # Create sequence
        sequence = self.create_inout_sequences(series, self.input_window)
        sequence = sequence[: -self.output_window]

        return sequence.to(self.device)

    def get_batch(self, source, i, batch_size):
        seq_len = min(batch_size, len(source) - 1 - i)
        data = source[i : i + seq_len]
        input = torch.stack(
            torch.stack([item[0] for item in data]).chunk(self.input_window, 1)
        )
        target = torch.stack(
            torch.stack([item[1] for item in data]).chunk(self.input_window, 1)
        )
        return input, target

    def train(self, train_data, epoch):
        self.model.train()  # Turn on the evaluation mode
        total_loss = 0.0
        start_time = time.time()

        for batch, i in enumerate(range(0, len(train_data) - 1, self.batch_size)):
            data, targets = self.get_batch(train_data, i, self.batch_size)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = max(1, int(len(train_data) / self.batch_size / 5))
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                logger.info(
                    f"| epoch {epoch:3d} | {batch:5d}/{len(train_data // self.batch_size):5d} batches |"
                )
                logger.info(
                    f"lr {self.scheduler.get_lr()[0]:02.10f} | {(elapsed * 1000 / log_interval):5.2f} ms |"
                )
                logger.info(f"loss {cur_loss:5.7f}")
                total_loss = 0
                start_time = time.time()

    def fit(self, X_train, y_train):
        # Convert train dataset to sequences
        logger.info("Converting train data into sequences...")
        train_data = self.convert_input(X_train)

        # Initialize model
        logger.info("Initializing model...")
        self.model = StockTransformer().to(self.device)

        # Initialize Loss function
        logger.info("Initializing loss function, optimizer and scheduler...")
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1.0, gamma=0.95
        )

        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            self.train(train_data, epoch)
            logger.info("-" * 80)
            logger.info(
                f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s".format()
            )
            logger.info("-" * 80)
            self.scheduler.step()

        return self.model

    def predict_proba(self, X_raw):
        # Convert X to sequences
        sequences = self.convert_input(X_raw)

        # Turn off gradients
        _ = self.model.eval()
        forecast_seq = torch.Tensor(0)
        with torch.no_grad():
            for i in range(0, len(sequences) - 1):
                data, target = self.get_batch(sequences, i, 1)
                output = self.model(data)
                forecast_seq = torch.cat((forecast_seq, output[-1].view(-1).cpu()), 0)

        # Convert predicted sequence to 1 / 0s (like y)
        forecast_seq = forecast_seq.detach().numpy()
        forecast_seq = pd.DataFrame(forecast_seq, columns=["y_pred"])
        forecast_seq["y_pred_next"] = forecast_seq["y_pred"].shift(-1)
        forecast_seq = forecast_seq.ffill()
        forecast_seq["target"] = forecast_seq["y_pred_next"] - forecast_seq["y_pred"]
        forecast_seq["target"] = (forecast_seq["target"] > 0).astype(int)
        forecast_seq["target"] = np.where(forecast_seq["target"] > 0, 0.55, 0)

        # Stack predictions
        predictions = [np.array([0.5, 0.5], dtype="float32")] * (self.input_window + 3)
        for idx, row in forecast_seq.iterrows():
            predictions.append(np.array([0.5, row["target"]], dtype="float32"))
        predictions = np.stack(predictions, axis=0)

        return predictions


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class StockTransformer(nn.Module):
    """
    Transformer for Stock price prediction
    """

    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(StockTransformer, self).__init__()
        self.model_type = "Transformer"

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=10, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
