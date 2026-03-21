"""Convolutional neural network classifier for MNIST."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    CNN_DROPOUT,
    CNN_FILTERS_1,
    CNN_FILTERS_2,
    CNN_HIDDEN,
    LEARNING_RATE,
    NUM_EPOCHS,
)
from src.interface import MnistClassifierInterface
from src.validators import validate_X, validate_hyperparams, validate_y


class _ConvNet(nn.Module):
    def __init__(
        self,
        filters_1: int = CNN_FILTERS_1,
        filters_2: int = CNN_FILTERS_2,
        hidden: int = CNN_HIDDEN,
        dropout: float = CNN_DROPOUT,
    ) -> None:
        """Initialize convolutional and classification layers.

        Params:
            filters_1:
                Number of filters in the first convolutional layer.
            filters_2:
                Number of filters in the second convolutional layer.
            hidden:
                Number of neurons in the fully connected hidden layer.
            dropout:
                Dropout probability.

        Returns:
            None
        """
        super().__init__()

        # Extract spatial features from input images
        self.features = nn.Sequential(
            nn.Conv2d(1, filters_1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters_1, filters_2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Map extracted features to class logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters_2 * 7 * 7, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Params:
            x:
                Input tensor of shape (N, 1, 28, 28).

        Returns:
            torch.Tensor:
                Raw class logits of shape (N, 10).
        """
        return self.classifier(self.features(x))


class CNNMnistClassifier(MnistClassifierInterface):
    """Train and run a compact CNN on MNIST."""

    def __init__(
        self,
        epochs: int = NUM_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        filters_1: int = CNN_FILTERS_1,
        filters_2: int = CNN_FILTERS_2,
        hidden: int = CNN_HIDDEN,
        dropout: float = CNN_DROPOUT,
    ) -> None:
        """Initialize the CNN classifier.

        Params:
            epochs:
                Number of training epochs.
            batch_size:
                Mini-batch size.
            lr:
                Learning rate.
            filters_1:
                Number of filters in the first convolutional layer.
            filters_2:
                Number of filters in the second convolutional layer.
            hidden:
                Number of neurons in the fully connected hidden layer.
            dropout:
                Dropout probability.

        Returns:
            None

        Raises:
            ValueError:
                If hyperparameters are invalid.
        """
        # Validate hyperparameters before model initialization
        validate_hyperparams(epochs, batch_size, lr)

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # Select device: GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and move it to the selected device
        self.model = _ConvNet(filters_1, filters_2, hidden, dropout).to(self.device)

        # Internal flag to prevent prediction before training
        self._trained = False

        print(f"[CNN] Device: {self.device}")

    @staticmethod
    def _preprocess(X: torch.Tensor) -> torch.Tensor:
        """Add channel dimension and normalize raw image tensors.

        Params:
            X:
                Input images of shape (N, 28, 28).

        Returns:
            torch.Tensor:
                Tensor of shape (N, 1, 28, 28) normalized to [0, 1].
        """
        # Convert images from (N, 28, 28) → (N, 1, 28, 28) and normalize pixel values
        return X.unsqueeze(1).float() / 255.0

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Fit the CNN with mini-batch gradient descent.

        Params:
            X_train:
                Training images of shape (N, 28, 28).
            y_train:
                Training labels of shape (N,).

        Returns:
            None

        Raises:
            TypeError:
                If input data has the wrong type.
            ValueError:
                If input data is invalid.
            RuntimeError:
                If training fails internally.
        """
        # Validate inputs before preprocessing
        validate_X(X_train)
        validate_y(y_train, X_train)

        # Prepare data for convolution layers
        X = self._preprocess(X_train)
        y = y_train.long()

        # Create mini-batches for stochastic optimization
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        try:
            self.model.train()

            for epoch in range(self.epochs):
                running = 0.0

                for X_batch, y_batch in loader:
                    # Move batch to the correct device (CPU/GPU)
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Standard training step:
                    # 1) zero gradients
                    # 2) forward pass
                    # 3) compute loss
                    # 4) backpropagation
                    # 5) update weights
                    optimizer.zero_grad()
                    loss = loss_fn(self.model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()

                    running += loss.item()

                print(
                    f"[CNN] Epoch {epoch + 1:>2}/{self.epochs} — "
                    f"loss: {running / len(loader):.4f}"
                )

        except RuntimeError as exc:
            # Add context and re-raise; logging is handled centrally in main.py
            raise RuntimeError(f"[CNN] Training failed: {exc}") from exc

        # Mark model as trained
        self._trained = True

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict digit labels for the given images.

        Params:
            X:
                Input images of shape (N, 28, 28).

        Returns:
            np.ndarray:
                Predicted class labels.

        Raises:
            TypeError:
                If input data has the wrong type.
            ValueError:
                If input data is invalid.
            RuntimeError:
                If prediction is requested before training or fails internally.
        """
        # Prevent inference before training
        if not self._trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train() before predict()."
            )

        # Validate input before preprocessing
        validate_X(X)

        try:
            self.model.eval()

            with torch.no_grad():
                # Preprocess input, run inference, and convert predictions to NumPy
                return (
                    self.model(self._preprocess(X).to(self.device))
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )

        except RuntimeError as exc:
            # Add context and re-raise; logging is handled in main.py
            raise RuntimeError(f"[CNN] Prediction failed: {exc}") from exc

    def save(self, path: Path) -> None:
        """Save the trained CNN model weights to disk.

        Params:
            path:
                File path to save the model to.

        Returns:
            None

        Raises:
            RuntimeError:
                If the model has not been trained yet.
        """
        if not self._trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train() before save()."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[CNN] Model saved to {path}")
