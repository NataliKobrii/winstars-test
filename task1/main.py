"""Run training and evaluation."""

import sys

from config import (
    BATCH_SIZE,
    DEBUG,
    LEARNING_RATE,
    LOG_FILE,
    MODEL_SAVE_DIR,
    NUM_EPOCHS,
)
from src.classifier import MnistClassifier
from src.data_loader import load_mnist
from src.logger import setup_logging
from src.utils import plot_accuracy_comparison, plot_confusion_matrix, print_accuracy

logger = setup_logging(__name__)


def format_error_message(base_message: str) -> str:
    """Append log-file details in debug mode.

    Parameters:
        base_message: str
            User-facing error message.

    Returns:
        str:
            Error message, extended with log-file information.
    """
    if DEBUG:
        return f"{base_message} See {LOG_FILE} for full traceback."
    return base_message


def main() -> None:
    """
    Run the full training and evaluation pipeline:
        1. Loads the MNIST dataset (train and test splits);
        2. Trains the following models:
            - Random Forest (RF);
            - Feed-Forward Neural Network (NN);
            - Convolutional Neural Network (CNN).
        3. Generates predictions on the test set for each model.
        4. Computes and prints accuracy metrics to the console.
        5. Aggregates results for comparison.
        6. Saves visualization outputs:
            - Accuracy comparison plot;
            - Confusion matrix (CNN).

    Parameters: None

    Returns: None

    Outputs:
    - Console accuracy summary (per model and final comparison);
    - Trained models saved to outputs/models/;
    - Accuracy comparison plot saved to disk;
    - Confusion matrix for CNN saved to disk.

    Raises:
        ImportError:
            If required dependencies are missing.
        OSError:
            If dataset or file operations fail.
        ValueError:
            If input data or configuration values are invalid.
        TypeError:
            If arguments have incorrect types.
        RuntimeError:
            If model training or prediction fails.
    """
    X_train, y_train, X_test, y_test = load_mnist()
    results: dict[str, float] = {}

    print("Train Random Forest:")
    rf = MnistClassifier(algorithm="rf")
    rf.train(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results["RF"] = print_accuracy("RF", y_test, rf_preds)
    rf.save(MODEL_SAVE_DIR / "rf_model.joblib")

    print("Train Feed-Forward Neural Network:")
    nn = MnistClassifier(
        algorithm="nn",
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )
    nn.train(X_train, y_train)
    nn_preds = nn.predict(X_test)
    results["NN"] = print_accuracy("NN", y_test, nn_preds)
    nn.save(MODEL_SAVE_DIR / "nn_model.pt")

    print("Train Convolutional Neural Network:")
    cnn = MnistClassifier(
        algorithm="cnn",
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )
    cnn.train(X_train, y_train)
    cnn_preds = cnn.predict(X_test)
    results["CNN"] = print_accuracy("CNN", y_test, cnn_preds)
    cnn.save(MODEL_SAVE_DIR / "cnn_model.pt")

    print("Final Results:")
    for name, acc in results.items():
        print(f"{name:<6}: {acc:.4f} ({acc * 100:.2f}%)")

    plot_accuracy_comparison(results, save=True)
    plot_confusion_matrix(y_test, cnn_preds, title="CNN Confusion Matrix", save=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        print("\n[Interrupted] Training stopped by user.", file=sys.stderr)
        raise SystemExit(1)
    except (ImportError, ValueError, TypeError, RuntimeError, OSError) as exc:
        logger.exception("Pipeline execution failed.")
        print(
            format_error_message(f"\n[Error] {type(exc).__name__}: {exc}."),
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception:
        logger.exception("Unexpected error occurred.")
        print(format_error_message("\n[Error] Unexpected failure."), file=sys.stderr)
        raise SystemExit(1)
