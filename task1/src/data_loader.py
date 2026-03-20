"""Load the MNIST dataset."""

import torch

from config import DATA_DIR
from src.logger import setup_logging

logger = setup_logging(__name__)


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download MNIST and return train/test tensors.

    Parameters:
        None

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            X_train, y_train, X_test, y_test

    Raises:
        ImportError:
            If torchvision is not available.
        OSError:
            If dataset files cannot be accessed.
        RuntimeError:
            If dataset shape is unexpected.
    """
    try:
        from torchvision import datasets
    except ImportError as exc:
        logger.exception("Failed to import torchvision.")
        raise ImportError("torchvision is required to load MNIST.") from exc

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        train_ds = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
        test_ds = datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    except OSError as exc:
        logger.exception("Could not access MNIST files in '%s'.", DATA_DIR)
        raise OSError(f"Could not read the MNIST dataset from '{DATA_DIR}'.") from exc
    except Exception:
        logger.exception("Failed to load MNIST dataset.")
        raise

    # Use raw tensors to keep a consistent (N, 28, 28) format across models
    X_train = train_ds.data
    y_train = train_ds.targets
    X_test = test_ds.data
    y_test = test_ds.targets

    if X_train.ndim != 3 or X_train.shape[1:] != (28, 28):
        raise RuntimeError(f"Unexpected training shape: {tuple(X_train.shape)}")

    if X_test.ndim != 3 or X_test.shape[1:] != (28, 28):
        raise RuntimeError(f"Unexpected test shape: {tuple(X_test.shape)}")

    print(f"Train: {tuple(X_train.shape)} | Test: {tuple(X_test.shape)}")

    return X_train, y_train, X_test, y_test
