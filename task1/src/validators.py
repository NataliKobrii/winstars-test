"""Shared validation helpers."""

import numpy as np
import torch

_EXPECTED_H = 28
_EXPECTED_W = 28


def validate_X(X: torch.Tensor) -> None:
    """Validate input images.

    Parameters:
        X: torch.Tensor
            Input images of shape (N, 28, 28).

    Returns:
        None

    Raises:
        TypeError:
            If X is not a torch.Tensor.
        ValueError:
            If X has an invalid shape or is empty.
    """
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor.")

    if X.ndim != 3 or X.shape[1] != _EXPECTED_H or X.shape[2] != _EXPECTED_W:
        raise ValueError("X must have shape (N, 28, 28).")

    if X.shape[0] == 0:
        raise ValueError("X cannot be empty.")


def validate_y(y: torch.Tensor, X: torch.Tensor) -> None:
    """Validate labels.

    Parameters:
        y: torch.Tensor
            Labels of shape (N,).
        X: torch.Tensor
            Input images used to check matching length.

    Returns:
        None

    Raises:
        TypeError:
            If y is not a torch.Tensor.
        ValueError:
            If y has an invalid shape or length.
    """
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor.")

    if y.ndim != 1:
        raise ValueError("y must have shape (N,).")

    if len(y) != len(X):
        raise ValueError("X and y must have the same length.")


def validate_y_pair(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate two label arrays.

    Parameters:
        y_true
            True labels.
        y_pred
            Predicted labels.

    Returns:
        None

    Raises:
        ValueError:
            If inputs are empty, not 1-dimensional, or lengths do not match.
    """
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")


def validate_hyperparams(epochs: int, batch_size: int, lr: float) -> None:
    """Validate neural network hyperparameters.

    Parameters:
        epochs: int
            Number of training epochs.
        batch_size: int
            Batch size.
        lr: float
            Learning rate.

    Returns:
        None

    Raises:
        ValueError:
            If a hyperparameter is invalid.
    """
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    if not isinstance(lr, (int, float)) or not (0 < lr <= 1):
        raise ValueError("learning rate (lr) must be a number in the range (0, 1].")


def validate_n_estimators(n_estimators: int) -> None:
    """Validate number of trees for Random Forest.

    Parameters:
        n_estimators: int
            Number of trees.

    Returns:
        None

    Raises:
        ValueError:
            If n_estimators is invalid.
    """
    if not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("n_estimators must be a positive integer.")
