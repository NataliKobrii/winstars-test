"""Train the NER model on synthetic animal data.

Generate BIO-tagged sentences (synthetic data with animal names)
and fine-tune a BERT model to recognize animal entities in text.
B - indicates the beginning of an entity.
I - indicates a token is contained inside the same entity (for example, the State token is a part of an entity like Empire State Building).
0 - indicates the token doesn’t correspond to any entity.
"""

import argparse
import sys

from config import NER_BATCH_SIZE, NER_EPOCHS, NER_LR
from src.logger import format_error_message, setup_logging

logger = setup_logging(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace:
            Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the animal NER model.")
    parser.add_argument(
        "--epochs", type=int, default=NER_EPOCHS, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=NER_BATCH_SIZE, help="Batch size."
    )
    parser.add_argument("--lr", type=float, default=NER_LR, help="Learning rate.")
    return parser.parse_args()


def main() -> None:
    """Generate synthetic data, train the NER model, and save it.

    Returns:
        None
    """
    args = parse_args()

    from src.models.ner_model import AnimalNERModel
    from src.ner_data_generator import generate_ner_dataset

    # Generate synthetic sentences with BIO tags, e.g. "There is a cow in the picture"
    # where "cow" is tagged as B-ANIMAL and the rest as O
    print("Generating synthetic NER training data...")
    train_data, val_data = generate_ner_dataset()
    print(f"Generated {len(train_data)} training + {len(val_data)} validation samples")

    # Initialize BERT for token classification (3 labels: O, B-ANIMAL, I-ANIMAL)
    ner = AnimalNERModel(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Fine-tune BERT on the synthetic data
    print("Training NER model...")
    ner.train(train_data, val_data)

    # Save trained model to load later in the pipeline
    ner.save()
    print("NER training complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        print("\n[Interrupted] Training stopped by user.", file=sys.stderr)
        raise SystemExit(1)
    except (ImportError, ValueError, TypeError, RuntimeError, OSError) as exc:
        logger.exception("NER training failed.")
        print(
            format_error_message(f"\n[Error] {type(exc).__name__}: {exc}."),
            file=sys.stderr,
        )
        raise SystemExit(1)
    except Exception:
        logger.exception("Unexpected error occurred.")
        print(format_error_message("\n[Error] Unexpected failure."), file=sys.stderr)
        raise SystemExit(1)
