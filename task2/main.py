"""Run the animal verification pipeline.

The main entry point for the whole task:
takes a text and image as input, runs NER to extract the animal name
from the text, classifies the image, and checks if they match.
Output is a single boolean value.
"""

import argparse
import sys
from pathlib import Path

from src.logger import format_error_message, setup_logging

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (according to https://docs.python.org/3/howto/argparse.html).

    Returns:
        argparse.Namespace:
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Verify if an image contains the animal mentioned in text."
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Input text describing an animal."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the image file."
    )
    return parser.parse_args()


def main() -> None:
    """Load the pipeline, verify the text-image pair, and print the result.

    Returns:
        None
    """
    args = parse_args()

    from src.pipeline import AnimalVerificationPipeline

    # Load both models (NER and image classifier)
    pipeline = AnimalVerificationPipeline()

    # Run the pipeline and print True or False
    result = pipeline.verify(args.text, Path(args.image))
    print(result)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        print("\n[Interrupted] Pipeline stopped by user.", file=sys.stderr)
        raise SystemExit(1)
    except (
        ImportError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        FileNotFoundError,
    ) as exc:
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
