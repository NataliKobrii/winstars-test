"""Run image classifier inference on an input image."""

import argparse
import sys

from src.logger import format_error_message, setup_logging

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace:
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Classify an animal image.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file."
    )
    return parser.parse_args()


def main() -> None:
    """Load the classifier and predict the animal class.

    Returns:
        None
    """
    args = parse_args()

    from pathlib import Path

    from src.models.image_classifier import AnimalImageClassifier

    # Create the classifier and load saved weights
    clf = AnimalImageClassifier()
    clf.load()

    # Classify the image and print the predicted animal name
    predicted_class = clf.predict(Path(args.image))
    print(predicted_class)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Inference interrupted by user.")
        print("\n[Interrupted] Inference stopped by user.", file=sys.stderr)
        raise SystemExit(1)
    except (
        ImportError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        FileNotFoundError,
    ) as exc:
        logger.exception("Classifier inference failed.")
        print(
            format_error_message(f"\n[Error] {type(exc).__name__}: {exc}."),
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception:
        logger.exception("Unexpected error occurred.")
        print(format_error_message("\n[Error] Unexpected failure."), file=sys.stderr)
        raise SystemExit(1)
