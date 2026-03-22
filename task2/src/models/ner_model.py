"""BERT-based Named Entity Recognition model for animal detection.

Uses HuggingFace's BertForTokenClassification to fine-tune BERT
for extracting animal names from text using BIO tagging.
B - indicates the beginning of an entity.
I - indicates a token is contained inside the same entity (for example, the State token is a part of an entity like Empire State Building).
0 - indicates the token doesn’t correspond to any entity.
See: https://huggingface.co/docs/transformers/tasks/token_classification
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertForTokenClassification, BertTokenizerFast

from config import (
    NER_BATCH_SIZE,
    NER_EPOCHS,
    NER_ID2LABEL,
    NER_LABEL2ID,
    NER_LR,
    NER_MAX_LENGTH,
    NER_MODEL_NAME,
)
from src.logger import setup_logging
from src.validators import validate_ner_hyperparams, validate_text

logger = setup_logging(__name__)


class _NERDataset(Dataset):
    """Prepares tokenized data for BERT token classification.

    BERT splits words into subword tokens (e.g. "elephant" -> ["ele", "##phant"]),
    but our BIO labels are per-word. So we need to map each subword back to its
    original word and assign the correct label.
    See: https://huggingface.co/docs/transformers/tasks/token_classification#preprocess
    """

    def __init__(
        self,
        data: list[dict],
        tokenizer: BertTokenizerFast,
        label2id: dict[str, int],
        max_length: int,
    ) -> None:
        self.encodings = []

        for sample in data:
            tokens = sample["tokens"]
            labels = sample["labels"]

            # is_split_into_words=True tells the tokenizer the input is already
            # split into words, so it only applies subword tokenization
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            # word_ids() maps each subword token back to the original word index
            # e.g. "There is a cow" -> ["[CLS]", "there", "is", "a", "cow", "[SEP]"]
            #                           word_ids: [None, 0, 1, 2, 3, None]
            # None means it's a special token, not a real word
            word_ids = encoding.word_ids(batch_index=0)
            aligned_labels = []

            for word_id in word_ids:
                if word_id is None:
                    # special tokens ([CLS], [SEP], [PAD]) get -100 which tells
                    # CrossEntropyLoss to ignore them (ignore_index=-100 by default).
                    # See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                    aligned_labels.append(-100)
                else:
                    aligned_labels.append(label2id[labels[word_id]])

            # Store the tokenized input for this sample:
            # - input_ids: token indices that BERT understands
            # - attention_mask: 1 for real tokens, 0 for padding (so BERT ignores padding)
            # - labels: the aligned BIO tags as integers
            # squeeze(0) removes the batch dimension since one sample is processed at a time
            self.encodings.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": torch.tensor(aligned_labels, dtype=torch.long),
                }
            )

    # __len__ and __getitem__ are required by PyTorch's Dataset class —
    # DataLoader uses them to know the size and fetch samples by index.
    # See: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> dict:
        return self.encodings[idx]


class AnimalNERModel:
    """Fine-tunes BERT to extract animal names from text.

    Uses BIO tagging (B-ANIMAL, I-ANIMAL, O) to identify animal entities.
    """

    def __init__(
        self,
        model_name: str = NER_MODEL_NAME,
        epochs: int = NER_EPOCHS,
        batch_size: int = NER_BATCH_SIZE,
        lr: float = NER_LR,
        max_length: int = NER_MAX_LENGTH,
    ) -> None:
        """Initialize the NER model.

        Parameters:
            model_name: str
                HuggingFace model identifier.
            epochs: int
                Number of training epochs.
            batch_size: int
                Batch size.
            lr: float
                Learning rate.
            max_length: int
                Maximum token sequence length.

        Returns:
            None

        Raises:
            ValueError:
                If hyperparameters are invalid.
        """
        validate_ner_hyperparams(epochs, batch_size, lr)

        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length

        # use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model and tokenizer are set to None here and only loaded
        # when needed — in train() or load(). This avoids downloading
        # the pretrained base model when only a saved one needs to be loaded.
        self.tokenizer = None
        self.model = None

        self._trained = False

        print(f"[NER] Device: {self.device}")

    def train(self, train_data: list[dict], val_data: list[dict]) -> list[float]:
        """Train the NER model on BIO-tagged token data.

        Parameters:
            train_data: list[dict]
                Training samples with 'tokens' and 'labels' keys.
            val_data: list[dict]
                Validation samples with 'tokens' and 'labels' keys.

        Returns:
            list[float]:
                Training loss per epoch.

        Raises:
            ValueError:
                If training data is empty.
            RuntimeError:
                If training fails.
        """
        if not train_data:
            raise ValueError("train_data must not be empty.")

        if not val_data:
            raise ValueError("val_data must not be empty.")

        # Load the tokenizer to know how to split text into tokens that BERT understands
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
        # Load pretrained BERT and replace the output layer with 3 labels (O, B-ANIMAL, I-ANIMAL)
        self.model = BertForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(NER_LABEL2ID),
            id2label=NER_ID2LABEL,
            label2id=NER_LABEL2ID,
        ).to(self.device)

        # Tokenize all samples and align BIO labels with subword tokens
        train_dataset = _NERDataset(
            train_data, self.tokenizer, NER_LABEL2ID, self.max_length
        )
        val_dataset = _NERDataset(
            val_data, self.tokenizer, NER_LABEL2ID, self.max_length
        )

        # DataLoader feeds samples to the model in batches instead of one by one
        # shuffle=True for training so the model doesn't memorize the order
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # AdamW is the Adam optimizer with weight decay, recommended for BERT fine-tuning
        # (BERT paper, Appendix A.3: https://arxiv.org/abs/1810.04805)
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        epoch_losses: list[float] = []

        try:
            # Put model in training mode
            self.model.train()

            for epoch in range(self.epochs):
                running = 0.0

                for batch in train_loader:
                    # Move data to GPU if available, otherwise stays on CPU
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    # Standard PyTorch training loop
                    # (see: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
                    optimizer.zero_grad()  # clear gradients from previous step
                    outputs = self.model(  # forward pass — model makes predictions
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,  # when labels are passed, BERT computes loss automatically
                    )
                    loss = outputs.loss
                    loss.backward()  # backward pass — compute gradients
                    optimizer.step()  # update weights based on gradients

                    running += loss.item()

                # Average training loss for this epoch
                avg_loss = running / len(train_loader)
                epoch_losses.append(avg_loss)

                # Check validation loss — if it starts going up while train loss goes down,
                # then the model is overfitting
                val_loss = self._evaluate_loss(val_loader)

                print(
                    f"[NER] Epoch {epoch + 1:>2}/{self.epochs} — "
                    f"train loss: {avg_loss:.4f} | validation loss: {val_loss:.4f}"
                )

        except RuntimeError as exc:
            raise RuntimeError(f"[NER] Training failed: {exc}") from exc

        self._trained = True
        return epoch_losses

    def _evaluate_loss(self, loader: DataLoader) -> float:
        """Compute average loss on a data loader.

        Parameters:
            loader: DataLoader
                Data loader to evaluate.

        Returns:
            float:
                Average loss.
        """
        # Switch to evaluation mode — disables dropout so predictions are consistent.
        # See: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval
        self.model.eval()
        total = 0.0

        # no_grad - according to PyTorch docs, gradients are not needed here;
        # only measuring the loss (not updating weights), so no need to track gradients
        # See: https://pytorch.org/docs/stable/generated/torch.no_grad.html
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Run forward pass and accumulate the loss
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total += outputs.loss.item()

        # Switch back to training mode for the next epoch
        self.model.train()
        # Return average loss across all validation batches
        return total / len(loader)

    def predict(self, text: str) -> list[str]:
        """Extract animal names from text.

        Parameters:
            text: str
                Input text string.

        Returns:
            list[str]:
                List of extracted animal names.

        Raises:
            TypeError:
                If text is not a string.
            ValueError:
                If text is empty.
            RuntimeError:
                If model has not been trained or loaded.
        """
        if not self._trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train() or load() first."
            )

        validate_text(text)

        # Slit text into words, then let BERT break them into subword tokens
        # e.g. "There is a cow" -> words: ["There", "is", "a", "cow"]
        # BERT then tokenizes: ["[CLS]", "there", "is", "a", "cow", "[SEP]"]
        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,  # tells tokenizer the input is already split into words
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",  # return PyTorch tensors
        )

        self.model.eval()

        # Run the model to get predictions for each token
        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Logits are raw scores for each label (O, B-ANIMAL, I-ANIMAL)
            # Argmax picks the one with the highest score
            predictions = outputs.logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        # BERT predicted labels for subword tokens, but labels per word are needed.
        # word_ids() maps subwords back to words,
        # and take only the first subword's prediction for each word
        word_ids = encoding.word_ids(batch_index=0)
        word_preds: dict[int, int] = {}

        for token_idx, word_id in enumerate(word_ids):
            # Skip special tokens (None) and already-seen words
            if word_id is not None and word_id not in word_preds:
                word_preds[word_id] = predictions[token_idx]

        # Reconstruct animal names from BIO tags
        # for example: words = ["There", "is", "a", "cow", "here"]
        #          tags  = [  O,      O,    O, B-ANIMAL,  O  ]
        #          result: ["cow"]
        animals: list[str] = []
        current_animal: list[str] = []

        for word_idx in range(len(words)):
            label_id = word_preds.get(word_idx, 0)
            label = NER_ID2LABEL.get(label_id, "O")

            if label == "B-ANIMAL":
                # Start of a new animal — save the previous one if any
                if current_animal:
                    animals.append(" ".join(current_animal))
                current_animal = [words[word_idx]]
            elif label == "I-ANIMAL" and current_animal:
                # Continuation of the current animal name
                current_animal.append(words[word_idx])
            else:
                # Not an animal token — save the current animal if one was being built
                if current_animal:
                    animals.append(" ".join(current_animal))
                    current_animal = []

        # If the sentence ends with an animal name (e.g. "I see a cow"),
        # the loop won't hit an O tag to save it, so it's saved here
        if current_animal:
            animals.append(" ".join(current_animal))

        return animals

    def save(self, directory: str | None = None) -> None:
        """Save the model and tokenizer.

        Parameters:
            directory: str | None
                Directory to save to. Uses NER_MODEL_DIR if None.

        Returns:
            None

        Raises:
            OSError:
                If saving fails.
        """
        from config import NER_MODEL_DIR

        save_dir = Path(directory) if directory else NER_MODEL_DIR
        save_dir.mkdir(
            parents=True, exist_ok=True
        )  # create the folder if it doesn't exist

        try:
            # save_pretrained is a HuggingFace method that saves the model weights
            # and tokenizer files so they can be loaded later without retraining
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
        except OSError as exc:
            logger.exception("Failed to save NER model to '%s'.", save_dir)
            raise OSError(f"Could not save NER model to '{save_dir}'.") from exc

        print(f"[NER] Model saved to {save_dir}")

    def load(self, directory: str | None = None) -> None:
        """Load a saved model and tokenizer.

        Parameters:
            directory: str | None
                Directory to load from. Uses NER_MODEL_DIR if None.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If the directory does not exist.
            OSError:
                If loading fails.
        """
        from config import NER_MODEL_DIR
        from pathlib import Path

        load_dir = Path(directory) if directory else NER_MODEL_DIR

        if not load_dir.exists():
            raise FileNotFoundError(f"NER model directory not found: {load_dir}")

        try:
            # Load the tokenizer and model weights that were saved earlier with save()
            self.tokenizer = BertTokenizerFast.from_pretrained(load_dir)
            self.model = BertForTokenClassification.from_pretrained(load_dir).to(
                self.device
            )
        except OSError as exc:
            logger.exception("Failed to load NER model from '%s'.", load_dir)
            raise OSError(f"Could not load NER model from '{load_dir}'.") from exc

        # Mark as trained so predict() knows the model is ready
        self._trained = True
        print(f"[NER] Model loaded from {load_dir}")
