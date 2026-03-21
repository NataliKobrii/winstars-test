"""Random Forest classifier for MNIST."""

from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError

from config import RANDOM_STATE, RF_N_ESTIMATORS
from src.interface import MnistClassifierInterface
from src.validators import validate_X, validate_n_estimators, validate_y


class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Initialize the Random Forest classifier.

        Params:
            n_estimators:
                Number of trees in the forest.
            random_state:
                Random seed used for reproducibility.

        Returns:
            None

        Raises:
            ValueError:
                If n_estimators is invalid.
        """
        # Validate hyperparameter before model creation
        validate_n_estimators(n_estimators)

        # Initialize sklearn Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        # Internal flag to prevent prediction before training
        self._trained = False

    @staticmethod
    def _preprocess(X: torch.Tensor) -> np.ndarray:
        """Flatten and normalize images for scikit-learn input.

        Params:
            X:
                Input images of shape (N, 28, 28).

        Returns:
            np.ndarray:
                Flattened image array of shape (N, 784).
        """
        # Convert images from (N, 28, 28) to (N, 784) and normalize to [0, 1]
        # Move to CPU and convert to NumPy, since sklearn does not support torch tensors
        return X.reshape(len(X), -1).cpu().numpy().astype(np.float32) / 255.0

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Fit the Random Forest model on MNIST images.

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
                If input data is invalid or sklearn rejects the data.
            RuntimeError:
                If training fails unexpectedly.
        """
        # Validate input tensors before processing
        validate_X(X_train)
        validate_y(y_train, X_train)

        print("[RF] Starting training...")

        # Preprocess images into flat feature vectors
        X_flat = self._preprocess(X_train)
        # Ensure tensor is on CPU and convert to NumPy array (required for scikit-learn models)
        y_labels = y_train.cpu().numpy()

        try:
            # Train Random Forest model on tabular representation of images
            self.model.fit(X_flat, y_labels)
        except ValueError as exc:
            raise ValueError(f"[RF] sklearn fitting failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"[RF] Training failed: {exc}") from exc

        self._trained = True  # Mark model as trained to allow prediction
        print(f"[RF] Training complete — {self.model.n_estimators} trees")

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
                If prediction is attempted before training or fails unexpectedly.
        """
        # Prevent inference before model is trained
        if not self._trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train() before predict()."
            )

        # Validate input before preprocessing
        validate_X(X)

        try:
            return self.model.predict(
                self._preprocess(X)
            )  # Preprocess input and generate predictions
        except NotFittedError as exc:
            raise RuntimeError(
                "[RF] Model is not fitted. Call train() before predict()."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"[RF] Prediction failed: {exc}") from exc

    def save(self, path: Path) -> None:
        """Save the trained Random Forest model to disk.

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
        joblib.dump(self.model, path)
        print(f"[RF] Model saved to {path}")
