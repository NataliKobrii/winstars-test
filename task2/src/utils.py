"""Evaluation and plotting helpers.

Used by train_classifier.py and demo.ipynb for metrics and visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from config import ANIMAL_CLASSES, FIGURE_SAVE_DIR
from src.logger import setup_logging
from src.validators import validate_y_pair

logger = setup_logging(__name__)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert input to a NumPy array.

    Parameters:
        x: torch.Tensor | np.ndarray
            Input tensor or array.

    Returns:
        np.ndarray:
            Converted NumPy array.
    """
    # Convert torch.Tensor to NumPy; otherwise ensure NumPy array
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    return np.asarray(x)


def _prepare_y_pair(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert and validate label arrays.

    Parameters:
        y_true: torch.Tensor | np.ndarray
            True labels.
        y_pred: torch.Tensor | np.ndarray
            Predicted labels.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Converted NumPy arrays.
    """
    # Convert both to numpy so sklearn can work with them
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    # Make sure arrays have matching shapes and are not empty
    validate_y_pair(y_true, y_pred)

    return y_true, y_pred


def evaluate(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
) -> float:
    """Compute classification accuracy.

    Parameters:
        y_true: torch.Tensor | np.ndarray
            True labels.
        y_pred: torch.Tensor | np.ndarray
            Predicted labels.

    Returns:
        float:
            Accuracy score.

    Raises:
        ValueError:
            If inputs are empty, have invalid shape, or lengths do not match.
    """
    y_true, y_pred = _prepare_y_pair(y_true, y_pred)

    try:
        return float(accuracy_score(y_true, y_pred))
    except ValueError as exc:
        raise ValueError("Could not compute accuracy.") from exc


def print_accuracy(
    name: str,
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
) -> float:
    """Print and return model accuracy.

    Parameters:
        name: str
            Model name.
        y_true: torch.Tensor | np.ndarray
            True labels.
        y_pred: torch.Tensor | np.ndarray
            Predicted labels.

    Returns:
        float:
            Accuracy score.

    Raises:
        ValueError:
            If name is empty or invalid.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string.")

    # Compute and print accuracy for the given model
    acc = evaluate(y_true, y_pred)
    print(f"[{name}] Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    return acc


def plot_confusion_matrix(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save: bool = False,
) -> None:
    """Plot and optionally save a confusion matrix.

    Parameters:
        y_true: torch.Tensor | np.ndarray
            True labels.
        y_pred: torch.Tensor | np.ndarray
            Predicted labels.
        class_names: list[str] | None
            Display labels. Uses ANIMAL_CLASSES if None.
        title: str
            Figure title.
        save: bool
            Whether to save the figure.

    Returns:
        None

    Raises:
        ValueError:
            If inputs or arguments are invalid.
        OSError:
            If saving the figure fails.
    """
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string.")
    if not isinstance(save, bool):
        raise ValueError("save must be a boolean value.")

    y_true, y_pred = _prepare_y_pair(y_true, y_pred)
    labels = class_names or ANIMAL_CLASSES

    try:
        cm = confusion_matrix(y_true, y_pred)
    except ValueError as exc:
        raise ValueError("Could not build confusion matrix.") from exc

    # Create a dedicated figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(14, 12))

    # Render confusion matrix with animal class labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", xticks_rotation=45)

    # Add title and adjust spacing
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save:
        FIGURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_SAVE_DIR / f"{title.replace(' ', '_')}.png"
        try:
            fig.savefig(path, dpi=120)
        except OSError as exc:
            logger.exception("Could not save confusion matrix figure to '%s'.", path)
            raise OSError("Could not save confusion matrix figure.") from exc
        print(f"Saved figure: {path}")

    plt.show()
    plt.close(fig)


def plot_training_history(
    losses: list[float],
    title: str = "Training Loss",
    save: bool = False,
) -> None:
    """Plot training loss over epochs.

    Parameters:
        losses: list[float]
            Loss values per epoch.
        title: str
            Figure title.
        save: bool
            Whether to save the figure.

    Returns:
        None

    Raises:
        ValueError:
            If losses is empty or arguments are invalid.
        OSError:
            If saving the figure fails.
    """
    if not losses:
        raise ValueError("losses must not be empty.")

    # Plot loss per epoch to see if the model is converging
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        FIGURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_SAVE_DIR / f"{title.replace(' ', '_')}.png"
        try:
            fig.savefig(path, dpi=120)
        except OSError as exc:
            logger.exception("Could not save training history figure to '%s'.", path)
            raise OSError("Could not save training history figure.") from exc
        print(f"Saved figure: {path}")

    plt.show()
    plt.close(fig)


def plot_class_distribution(
    labels: list[str],
    title: str = "Class Distribution",
    save: bool = False,
) -> None:
    """Plot the distribution of classes in the dataset.

    Parameters:
        labels: list[str]
            Class labels for each sample.
        title: str
            Figure title.
        save: bool
            Whether to save the figure.

    Returns:
        None

    Raises:
        ValueError:
            If labels is empty.
        OSError:
            If saving the figure fails.
    """
    if not labels:
        raise ValueError("labels must not be empty.")

    # Count how many images per class
    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(unique, counts, color="#4878CF", edgecolor="white")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save:
        FIGURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_SAVE_DIR / f"{title.replace(' ', '_')}.png"
        try:
            fig.savefig(path, dpi=120)
        except OSError as exc:
            logger.exception("Could not save class distribution figure to '%s'.", path)
            raise OSError("Could not save class distribution figure.") from exc
        print(f"Saved figure: {path}")

    plt.show()
    plt.close(fig)


def show_sample_predictions(
    image_paths: list,
    y_true: list[str],
    y_pred: list[str],
    model_name: str = "Classifier",
    n_correct: int = 5,
    n_wrong: int = 5,
) -> None:
    """Show sample correct and incorrect predictions.

    Parameters:
        image_paths: list
            Paths to the image files.
        y_true: list[str]
            True class labels.
        y_pred: list[str]
            Predicted class labels.
        model_name: str
            Name shown in the figure title.
        n_correct: int
            Number of correct predictions to display.
        n_wrong: int
            Number of incorrect predictions to display.

    Returns:
        None
    """
    from PIL import Image

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Find which predictions were correct and which were wrong
    correct_idx = np.where(y_true_arr == y_pred_arr)[0]
    wrong_idx = np.where(y_true_arr != y_pred_arr)[0]

    # Only take the first n samples of each
    correct_idx = correct_idx[:n_correct]
    wrong_idx = wrong_idx[:n_wrong]

    # Build a 2-row grid: top row = correct (green), bottom row = wrong (red)
    cols = max(n_correct, n_wrong)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 7))

    # Show correct predictions in the top row
    for i in range(cols):
        ax = axes[0, i]
        if i < len(correct_idx):
            idx = correct_idx[i]
            img = Image.open(image_paths[idx]).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"T:{y_true[idx]}\nP:{y_pred[idx]}", fontsize=8, color="green")
        ax.axis("off")

    # Show wrong predictions in the bottom row
    for i in range(cols):
        ax = axes[1, i]
        if i < len(wrong_idx):
            idx = wrong_idx[i]
            img = Image.open(image_paths[idx]).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"T:{y_true[idx]}\nP:{y_pred[idx]}", fontsize=8, color="red")
        ax.axis("off")

    axes[0, 0].set_ylabel("Correct", fontsize=11)
    axes[1, 0].set_ylabel("Wrong", fontsize=11)
    fig.suptitle(f"{model_name} — Sample Predictions", fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
