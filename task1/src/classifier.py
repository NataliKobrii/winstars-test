"""Unified wrapper for MNIST classification models."""

from pathlib import Path

import numpy as np
import torch

from src.interface import MnistClassifierInterface
from src.models.cnn_classifier import CNNMnistClassifier
from src.models.nn_classifier import FeedForwardMnistClassifier
from src.models.rf_classifier import RandomForestMnistClassifier


class MnistClassifier:
    """
    Unified MNIST classifier wrapper.
    """

    # Maps algorithm name to its implementation class
    _REGISTRY: dict[str, type[MnistClassifierInterface]] = {
        "rf": RandomForestMnistClassifier,
        "nn": FeedForwardMnistClassifier,
        "cnn": CNNMnistClassifier,
    }

    def __init__(self, algorithm: str, **kwargs) -> None:
        """Initialize the selected model.

        Parameters:
            algorithm: str
                Model type: "rf", "nn", or "cnn".
            **kwargs:
                Optional model-specific hyperparameters.

        Returns:
            None

        Raises:
            TypeError:
                If algorithm is not a string or a model parameter has the wrong type.
            ValueError:
                If algorithm is not supported or a model parameter is invalid.
        """
        if not isinstance(algorithm, str):
            raise TypeError("algorithm must be a string.")

        if algorithm not in self._REGISTRY:
            raise ValueError("algorithm must be one of: 'rf', 'nn', 'cnn'.")

        self.algorithm = algorithm
        self._model: MnistClassifierInterface = self._REGISTRY[algorithm](**kwargs)

    def __repr__(self) -> str:
        return f"MnistClassifier(algorithm='{self.algorithm}')"

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Train the selected model.

        Parameters:
            X_train: torch.Tensor
                Training images of shape (N, 28, 28).
            y_train: torch.Tensor
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
        self._model.train(X_train, y_train)

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predict labels for input images.

        Parameters:
            X: torch.Tensor
                Input images of shape (N, 28, 28).

        Returns:
            np.ndarray:
                Predicted labels of shape (N,).

        Raises:
            TypeError:
                If input data has the wrong type.
            ValueError:
                If input data is invalid.
            RuntimeError:
                If prediction is requested before training or fails internally.
        """
        return self._model.predict(X)

    def save(self, path: Path) -> None:
        """Save the trained model to disk.

        Parameters:
            path: Path
                File path to save the model to.

        Returns:
            None

        Raises:
            RuntimeError:
                If the model has not been trained yet.
        """
        self._model.save(path)
