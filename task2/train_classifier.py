"""Train the image classifier on the animal dataset.

Downloads the Kaggle animal dataset (if not already cached),
fine-tunes a pretrained ResNet-50 on 15 animal classes,
evaluates on a held-out test set, and saves the model + plots.
"""

import argparse
import sys

import numpy as np

from config import CLF_BATCH_SIZE, CLF_EPOCHS, CLF_LR
from src.logger import format_error_message, setup_logging

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace:
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the animal image classifier.")
    parser.add_argument(
        "--epochs", type=int, default=CLF_EPOCHS, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=CLF_BATCH_SIZE, help="Batch size."
    )
    parser.add_argument("--lr", type=float, default=CLF_LR, help="Learning rate.")
    return parser.parse_args()


def main() -> None:
    """Download dataset, train classifier, evaluate, and save.

    Returns:
        None
    """
    args = parse_args()

    from src.data_loader import get_train_val_test_split, load_animal_dataset
    from src.models.image_classifier import AnimalImageClassifier
    from src.utils import (
        plot_confusion_matrix,
        plot_training_history,
        print_accuracy,
    )

    # Load images from Kaggle dataset (downloads automatically if not cached)
    print("Loading dataset...")
    image_paths, labels = load_animal_dataset()

    # Split into training/validation/test with stratification to keep class balance
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = (
        get_train_val_test_split(image_paths, labels)
    )

    # Use pretrained ResNet-50, replace last fully connected layer for 15 animal classes
    clf = AnimalImageClassifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Fine-tune on training data, validate each epoch
    print("Training classifier...")
    losses = clf.train(train_paths, train_labels, val_paths, val_labels)

    # Check how well it does on images it hasn't seen during training
    print("Evaluating on test set...")
    test_preds = [clf.predict(p) for p in test_paths]
    print_accuracy("Classifier", np.array(test_labels), np.array(test_preds))

    # Save model weights to load later in the pipeline
    clf.save()

    # Save training loss curve and confusion matrix as images
    plot_training_history(losses, title="Classifier_Training_Loss", save=True)
    plot_confusion_matrix(
        np.array(test_labels),
        np.array(test_preds),
        title="Classifier_Confusion_Matrix",
        save=True,
    )

    print("Classifier training complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        print("\n[Interrupted] Training stopped by user.", file=sys.stderr)
        raise SystemExit(1)
    except (ImportError, ValueError, TypeError, RuntimeError, OSError) as exc:
        logger.exception("Classifier training failed.")
        print(
            format_error_message(f"\n[Error] {type(exc).__name__}: {exc}."),
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception:
        logger.exception("Unexpected error occurred.")
        print(format_error_message("\n[Error] Unexpected failure."), file=sys.stderr)
        raise SystemExit(1)
