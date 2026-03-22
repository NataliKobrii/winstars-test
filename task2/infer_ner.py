"""Run NER inference on input text."""

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
    parser = argparse.ArgumentParser(description="Extract animal names from text.")
    parser.add_argument(
        "--text", type=str, required=True, help="Input text to analyze."
    )
    return parser.parse_args()


def main() -> None:
    """Load the NER model and extract animal names from text.

    Returns:
        None
    """
    args = parse_args()

    from src.models.ner_model import AnimalNERModel

    ner = AnimalNERModel()
    ner.load()

    animals = ner.predict(args.text)
    print(animals)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Inference interrupted by user.")
        print("\n[Interrupted] Inference stopped by user.", file=sys.stderr)
        raise SystemExit(1)
    except (ImportError, ValueError, TypeError, RuntimeError, OSError) as exc:
        logger.exception("NER inference failed.")
        print(
            format_error_message(f"\n[Error] {type(exc).__name__}: {exc}."),
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception:
        logger.exception("Unexpected error occurred.")
        print(format_error_message("\n[Error] Unexpected failure."), file=sys.stderr)
        raise SystemExit(1)
