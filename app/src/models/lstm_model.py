import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger()


class LSTMModel(ClassifierMixin, BaseEstimator):
    """
    Wrapper for a PyTorch LSTM model for next step prediction
    """

    def __init__(
        self,
        window_size=30,
        hidden_size=64,
        num_layers=2,
        batch_size=32,
        lr=0.001,
        num_epochs=10,
        verbose=0,
    ) -> None:
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.model = None
        self.lr = lr
        self.num_epochs = num_epochs
        self.verbose = verbose

    def convert_to_dataloader(self, X_raw, y_raw=None, shuffle: bool = True):
        """
        Adapter to be used inside NN models
        """
        logger.info("Converting numpy arrays to TensorDataset and DataLoader...")
        # Convert to PyTorch Tensors
        X, y = [], []
        for i in range(len(X_raw) - self.window_size - 1):
            X.append(X_raw[i : i + self.window_size])
            if y_raw is not None:
                y.append(y_raw[i + self.window_size])
            else:
                y.append(0)

        # Now convert lists to tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        # DataLoader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        return dataloader

    def train_model(self, dataloader) -> None:
        """
        NN model training loop
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Switch model to train mode
        self.model.train()
        for epoch in range(self.num_epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}")

    def fit(self, X_train, y_train):
        # Convert data
        dataloader = self.convert_to_dataloader(X_raw=X_train, y_raw=y_train)
        # Initialize model
        logger.info("Initializing model...")
        self.model = StockLSTM(
            n_features=X_train.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )
        # Train it
        logger.info("Training model...")
        self.train_model(dataloader)
        return self.model

    def predict_proba(self, X_raw):
        dataloader = self.convert_to_dataloader(
            X_raw, shuffle=False
        )  # !!! turn off shuffle to keep prediction indices correct

        _ = self.model.eval()
        predictions = [np.array([0.5, 0.5], dtype="float32")] * (
            self.window_size + 1
        )  # first ticks have no real prediction
        with torch.no_grad():
            for inputs, _ in dataloader:
                logits = self.model(inputs)
                predictions.extend(logits.numpy())

        # Convert to a single np.array
        predictions = np.stack(predictions, axis=0)

        return predictions


class StockLSTM(nn.Module):
    """
    Neural Network with LSTM core
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)
        self.sm = nn.Softmax(1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.linear(h_n[-1])
        probabilities = self.sm(x)
        return probabilities
