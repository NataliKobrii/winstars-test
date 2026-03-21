"""Feed-forward neural network classifier for MNIST."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NN_DROPOUT,
    NN_HIDDEN_1,
    NN_HIDDEN_2,
    NUM_EPOCHS,
)
from src.interface import MnistClassifierInterface
from src.validators import validate_X, validate_hyperparams, validate_y


class _FeedForwardNet(nn.Module):
    def __init__(
        self,
        hidden_1: int = NN_HIDDEN_1,
        hidden_2: int = NN_HIDDEN_2,
        dropout: float = NN_DROPOUT,
    ) -> None:
        """Initialize dense layers of the network.

        Params:
            hidden_1:
                Number of neurons in the first hidden layer.
            hidden_2:
                Number of neurons in the second hidden layer.
            dropout:
                Dropout probability.

        Returns:
            None
        """
        super().__init__()

        # Define a simple multi-layer perceptron with dropout regularization
        self.network = nn.Sequential(
            nn.Linear(784, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Params:
            x:
                Input tensor of shape (N, 784).

        Returns:
            torch.Tensor:
                Raw class logits of shape (N, 10).
        """
        return self.network(x)


class FeedForwardMnistClassifier(MnistClassifierInterface):
    """Train and run a simple feed-forward network on MNIST."""

    def __init__(
        self,
        epochs: int = NUM_EPOCHS,
        batch_size: int = BATCH_SIZE,
        lr: float = LEARNING_RATE,
        hidden_1: int = NN_HIDDEN_1,
        hidden_2: int = NN_HIDDEN_2,
        dropout: float = NN_DROPOUT,
    ) -> None:
        """Initialize the feed-forward classifier.

        Params:
            epochs:
                Number of training epochs.
            batch_size:
                Batch size.
            lr:
                Learning rate.
            hidden_1:
                Number of neurons in the first hidden layer.
            hidden_2:
                Number of neurons in the second hidden layer.
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
        self.model = _FeedForwardNet(hidden_1, hidden_2, dropout).to(self.device)

        # Internal flag to prevent prediction before training
        self._trained = False

        print(f"[NN] Device: {self.device}")

    @staticmethod
    def _preprocess(X: torch.Tensor) -> torch.Tensor:
        """Flatten and normalize raw image tensors.

        Params:
            X:
                Input images of shape (N, 28, 28).

        Returns:
            torch.Tensor:
                Flattened tensor of shape (N, 784) normalized to [0, 1].
        """
        # Convert images from (N, 28, 28) → (N, 784) and normalize pixel values
        return X.reshape(len(X), -1).float() / 255.0

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Fit the network with mini-batch gradient descent.

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

        # Prepare data for the network
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
                    f"[NN] Epoch {epoch + 1:>2}/{self.epochs} — "
                    f"loss: {running / len(loader):.4f}"
                )

        except RuntimeError as exc:
            raise RuntimeError(f"[NN] Training failed: {exc}") from exc

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
            raise RuntimeError(f"[NN] Prediction failed: {exc}") from exc

    def save(self, path: Path) -> None:
        """Save the trained NN model weights to disk.

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
        print(f"[NN] Model saved to {path}")
