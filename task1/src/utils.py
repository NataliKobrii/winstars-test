"""Evaluation and plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from config import FIGURE_SAVE_DIR
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
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

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

    # Accumulate and print accuracy for all models
    acc = evaluate(y_true, y_pred)
    print(f"[{name}] Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    return acc


def plot_confusion_matrix(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    title: str = "Confusion Matrix",
    save: bool = False,
) -> None:
    """Plot and optionally save a confusion matrix.

    Parameters:
        y_true: torch.Tensor | np.ndarray
            True labels.
        y_pred: torch.Tensor | np.ndarray
            Predicted labels.
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

    try:
        cm = confusion_matrix(y_true, y_pred)
    except ValueError as exc:
        raise ValueError("Could not build confusion matrix.") from exc

    # Create a dedicated figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))

    # Render confusion matrix with class labels 0-9
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

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


def show_sample_predictions(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_correct: int = 5,
    n_wrong: int = 5,
) -> None:
    """Show sample correct and incorrect predictions side by side.

    Parameters:
        X: np.ndarray
            Test images of shape (N, 28, 28).
        y_true: np.ndarray
            True labels of shape (N,).
        y_pred: np.ndarray
            Predicted labels of shape (N,).
        model_name: str
            Name shown in the figure title.
        n_correct: int
            Number of correct predictions to display.
        n_wrong: int
            Number of incorrect predictions to display.

    Returns:
        None
    """
    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx = np.where(y_true != y_pred)[0]

    correct_idx = correct_idx[:n_correct]
    wrong_idx = wrong_idx[:n_wrong]

    cols = max(n_correct, n_wrong)
    fig, axes = plt.subplots(2, cols, figsize=(2.2 * cols, 5))

    for i in range(cols):
        ax = axes[0, i]
        if i < len(correct_idx):
            idx = correct_idx[i]
            ax.imshow(X[idx], cmap="gray")
            ax.set_title(f"T:{y_true[idx]} P:{y_pred[idx]}", fontsize=9, color="green")
        ax.axis("off")

    for i in range(cols):
        ax = axes[1, i]
        if i < len(wrong_idx):
            idx = wrong_idx[i]
            ax.imshow(X[idx], cmap="gray")
            ax.set_title(f"T:{y_true[idx]} P:{y_pred[idx]}", fontsize=9, color="red")
        ax.axis("off")

    axes[0, 0].set_ylabel("Correct", fontsize=11)
    axes[1, 0].set_ylabel("Wrong", fontsize=11)
    fig.suptitle(f"{model_name} — Sample Predictions", fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_accuracy_comparison(results: dict[str, float], save: bool = False) -> None:
    """Plot model accuracy comparison.

    Parameters:
        results: dict[str, float]
            Mapping of model names to accuracy scores.
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
    if not isinstance(results, dict):
        raise ValueError("results must be a dictionary mapping model names to scores.")
    if not isinstance(save, bool):
        raise ValueError("save must be a boolean value.")
    if not results:
        raise ValueError("results cannot be empty.")

    for key, value in results.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Model names must be non-empty strings.")
        if not isinstance(value, (int, float)):
            raise ValueError("Accuracy values must be numeric.")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("Accuracy values must be in range [0.0, 1.0].")

    names = list(results.keys())
    accs = [float(value) for value in results.values()]
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        names, accs, color=colors[: len(names)], width=0.45, edgecolor="white"
    )

    # Show accuracy values inside the bars
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() - 0.006, 0.02),
            f"{acc:.4f}",
            ha="center",
            va="top",
            color="white",
            fontweight="bold",
            fontsize=11,
        )

    # Keep the y-axis tight enough to emphasize small differences
    upper_limit = max(max(accs) + 0.02, 1.0)
    lower_limit = max(0.0, min(accs) - 0.05)
    ax.set_ylim(lower_limit, upper_limit)
    ax.set_title("Model accuracy on MNIST test set", fontsize=13)
    ax.set_ylabel("Accuracy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        FIGURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_SAVE_DIR / "accuracy_comparison.png"
        try:
            fig.savefig(path, dpi=120)
        except OSError as exc:
            logger.exception("Could not save accuracy comparison figure to '%s'.", path)
            raise OSError("Could not save accuracy comparison figure.") from exc
        print(f"Saved figure: {path}")

    plt.show()
    plt.close(fig)
