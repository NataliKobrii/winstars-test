"""Shared validation helpers."""

from pathlib import Path

import numpy as np


def validate_text(text: str) -> None:
    """Validate text input for NER.

    Parameters:
        text: str
            Input text string.

    Returns:
        None

    Raises:
        TypeError:
            If text is not a string.
        ValueError:
            If text is empty or blank.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not text.strip():
        raise ValueError("text cannot be empty or blank.")


def validate_image_path(image_path: Path) -> None:
    """Validate an image file path.

    Parameters:
        image_path: Path
            Path to an image file.

    Returns:
        None

    Raises:
        TypeError:
            If image_path is not a Path or str.
        FileNotFoundError:
            If the file does not exist.
        ValueError:
            If the file extension is not a supported image format.
    """
    if not isinstance(image_path, (str, Path)):
        raise TypeError("image_path must be a Path or str.")

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if image_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Unsupported image format '{image_path.suffix}'. "
            f"Supported: {sorted(valid_extensions)}"
        )


def validate_ner_hyperparams(epochs: int, batch_size: int, lr: float) -> None:
    """Validate NER training hyperparameters.

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


def validate_clf_hyperparams(
    epochs: int, batch_size: int, lr: float, image_size: int
) -> None:
    """Validate classifier training hyperparameters.

    Parameters:
        epochs: int
            Number of training epochs.
        batch_size: int
            Batch size.
        lr: float
            Learning rate.
        image_size: int
            Input image size (height and width).

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

    if not isinstance(image_size, int) or image_size < 1:
        raise ValueError("image_size must be a positive integer.")


def validate_y_pair(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate two label arrays.

    Parameters:
        y_true: np.ndarray
            True labels.
        y_pred: np.ndarray
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
