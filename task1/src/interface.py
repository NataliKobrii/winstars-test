"""Abstract base class for MNIST classification models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch


class MnistClassifierInterface(ABC):
    """
    Base interface for MNIST classification models.

    All models:
        - receive input tensors of shape (N, 28, 28)
        - handle their own preprocessing
        - return predictions as NumPy arrays of shape (N,)
    """

    @abstractmethod
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        Train the model.

        Parameters:
            X_train: torch.Tensor
                Training images of shape (N, 28, 28).
            y_train: torch.Tensor
                Labels of shape (N,).

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict labels for input images.

        Parameters:
            X: torch.Tensor
                Input images of shape (N, 28, 28).

        Returns:
            np.ndarray:
                Predicted labels of shape (N,).
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.

        Parameters:
            path: Path
                File path to save the model to.

        Returns:
            None
        """
        pass
