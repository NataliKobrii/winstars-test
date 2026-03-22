"""ResNet-50 image classifier for animal recognition."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import (
    ANIMAL_CLASSES,
    CLASSIFIER_MODEL_DIR,
    CLF_BATCH_SIZE,
    CLF_EPOCHS,
    CLF_IMAGE_SIZE,
    CLF_LR,
)
from src.logger import setup_logging
from src.validators import validate_clf_hyperparams, validate_image_path

logger = setup_logging(__name__)


class _AnimalDataset(Dataset):
    """PyTorch Dataset that loads images from file paths on demand.

    Instead of loading all images into memory at once, it opens each image
    only when DataLoader requests it.
    Extends torch.utils.data.Dataset (requires __len__ and __getitem__).
    See: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = (
            transform  # image preprocessing (resize, normalize, augmentation)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # Open image and convert to RGB (some images might be grayscale or RGBA)
        image = Image.open(self.image_paths[idx]).convert("RGB")
        # Apply transforms (resize, normalize, augmentation if training)
        image = self.transform(image)
        return image, self.labels[idx]


class AnimalImageClassifier:
    """ResNet-50 based image classifier for animal recognition.
    Fine-tunes a pretrained ResNet-50 model on 15 animal classes,
    replacing the final fully connected layer.
    """

    def __init__(
        self,
        epochs: int = CLF_EPOCHS,
        batch_size: int = CLF_BATCH_SIZE,
        lr: float = CLF_LR,
        image_size: int = CLF_IMAGE_SIZE,
        class_names: list[str] | None = None,
    ) -> None:
        """Initialize the image classifier.

        Parameters:
            epochs: int
                Number of training epochs.
            batch_size: int
                Batch size.
            lr: float
                Learning rate.
            image_size: int
                Input image size (height and width).
            class_names: list[str] | None
                List of class names. Uses ANIMAL_CLASSES if None.

        Returns:
            None

        Raises:
            ValueError:
                If hyperparameters are invalid.
        """
        validate_clf_hyperparams(epochs, batch_size, lr, image_size)

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.image_size = image_size
        self.class_names = class_names or ANIMAL_CLASSES
        self.num_classes = len(self.class_names)

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ResNet-50 with weights pretrained on ImageNet
        # See: https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # The original ResNet-50 was trained to classify 1000 ImageNet categories,
        # but only 15 animals are needed, so the last layer is swapped to match the classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)

        self._trained = False

        print(f"[Classifier] Device: {self.device}")

    def _get_train_transform(self) -> transforms.Compose:
        """Build training image transforms with augmentation.

        Returns:
            transforms.Compose:
                Training transforms pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size)
                ),  # resize to 224x224
                transforms.RandomHorizontalFlip(),  # randomly flip left-right
                transforms.RandomRotation(15),  # randomly rotate up to 15 degrees
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2
                ),  # vary brightness/contrast
                transforms.ToTensor(),  # convert image to PyTorch tensor
                transforms.Normalize(  # normalize with ImageNet statistics
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],  # (must match what ResNet was trained on)
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _get_val_transform(self) -> transforms.Compose:
        """Build validation/inference image transforms.

        Returns:
            transforms.Compose:
                Validation transforms pipeline.
        """
        # No augmentation here — during validation and inference the real image
        # is needed, not a modified version. Just resize and normalize.
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _label_to_index(self, labels: list[str]) -> list[int]:
        """Convert string labels to integer indices.

        Parameters:
            labels: list[str]
                Class name labels.

        Returns:
            list[int]:
                Integer indices.
        """
        name_to_idx = {name: i for i, name in enumerate(self.class_names)}
        return [name_to_idx[label] for label in labels]

    def train(
        self,
        train_paths: list[Path],
        train_labels: list[str],
        val_paths: list[Path],
        val_labels: list[str],
    ) -> list[float]:
        """Train the classifier on animal images.

        Parameters:
            train_paths: list[Path]
                Training image file paths.
            train_labels: list[str]
                Training class labels.
            val_paths: list[Path]
                Validation image file paths.
            val_labels: list[str]
                Validation class labels.

        Returns:
            list[float]:
                Training loss per epoch.

        Raises:
            ValueError:
                If training data is empty.
            RuntimeError:
                If training fails.
        """
        if not train_paths:
            raise ValueError("train_paths must not be empty.")

        if not val_paths:
            raise ValueError("val_paths must not be empty.")

        # Convert string labels ("cat", "dog") to integer indices (2, 4)
        # because CrossEntropyLoss needs integers, not strings
        train_indices = self._label_to_index(train_labels)
        val_indices = self._label_to_index(val_labels)

        # Wrap data in PyTorch Datasets — training gets augmentation, validation does not
        train_dataset = _AnimalDataset(
            train_paths, train_indices, self._get_train_transform()
        )
        val_dataset = _AnimalDataset(val_paths, val_indices, self._get_val_transform())

        # DataLoader feeds images to the model in batches (32 at a time)
        # shuffle=True randomizes the order each epoch so the model learns better
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # Adam optimizer (see: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # CrossEntropyLoss is standard for multi-class classification
        loss_fn = nn.CrossEntropyLoss()
        epoch_losses: list[float] = []

        try:
            for epoch in range(self.epochs):
                self.model.train()  # enable training mode
                running = 0.0

                for images, targets in train_loader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)

                    # Standard training loop (same as NER, just different model)
                    optimizer.zero_grad()  # clear previous gradients
                    outputs = self.model(images)  # forward pass
                    loss = loss_fn(outputs, targets)  # compute loss
                    loss.backward()  # compute gradients
                    optimizer.step()  # update weights

                    running += loss.item()

                avg_loss = running / len(train_loader)
                epoch_losses.append(avg_loss)

                # Check validation accuracy after each epoch
                val_acc = self._evaluate_accuracy(val_loader)

                print(
                    f"[Classifier] Epoch {epoch + 1:>2}/{self.epochs} — "
                    f"loss: {avg_loss:.4f} | validation accuracy: {val_acc:.4f}"
                )

        except RuntimeError as exc:
            raise RuntimeError(f"[Classifier] Training failed: {exc}") from exc

        self._trained = True
        return epoch_losses

    def _evaluate_accuracy(self, loader: DataLoader) -> float:
        """Compute accuracy on a data loader.

        Parameters:
            loader: DataLoader
                Data loader to evaluate.

        Returns:
            float:
                Accuracy score.
        """
        self.model.eval()  # switch to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # no gradients needed for evaluation
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                # Argmax picks the class with the highest score for each image
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        return correct / total if total > 0 else 0.0

    def predict(self, image_path: Path) -> str:
        """Predict the animal class for a single image.

        Parameters:
            image_path: Path
                Path to the image file.

        Returns:
            str:
                Predicted class name.

        Raises:
            TypeError:
                If image_path is not a valid path.
            FileNotFoundError:
                If the image file does not exist.
            RuntimeError:
                If model has not been trained or loaded.
        """
        if not self._trained:
            raise RuntimeError(
                "Model has not been trained yet. Call train() or load() first."
            )

        image_path = Path(image_path)
        validate_image_path(image_path)

        # Preprocess the image the same way as during training (without augmentation)
        transform = self._get_val_transform()
        image = Image.open(image_path).convert("RGB")
        # unsqueeze(0) adds a batch dimension — model expects a batch, not a single image
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(image_tensor)
            # Pick the class with the highest score
            pred_idx = outputs.argmax(dim=1).item()

        # Convert integer index back to class name (e.g. 3 - "cow")
        return self.class_names[pred_idx]

    def save(self, directory: str | None = None) -> None:
        """Save model weights and class names.

        Parameters:
            directory: str | None
                Directory to save to. Uses CLASSIFIER_MODEL_DIR if None.

        Returns:
            None

        Raises:
            OSError:
                If saving fails.
        """
        save_dir = Path(directory) if directory else CLASSIFIER_MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save model weights
            torch.save(self.model.state_dict(), save_dir / "model.pth")
            # Save the class names to understand which index maps to which animal
            with open(save_dir / "class_names.json", "w") as f:
                json.dump(self.class_names, f)

        except OSError as exc:
            logger.exception("Failed to save classifier to '%s'.", save_dir)
            raise OSError(f"Could not save classifier to '{save_dir}'.") from exc

        print(f"[Classifier] Model saved to {save_dir}")

    def load(self, directory: str | None = None) -> None:
        """Load saved model weights and class names.

        Parameters:
            directory: str | None
                Directory to load from. Uses CLASSIFIER_MODEL_DIR if None.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If the directory or required files do not exist.
            OSError:
                If loading fails.
        """
        load_dir = Path(directory) if directory else CLASSIFIER_MODEL_DIR

        if not load_dir.exists():
            raise FileNotFoundError(f"Classifier directory not found: {load_dir}")

        try:
            # First load the class names to know how many output classes are needed
            with open(load_dir / "class_names.json") as f:
                self.class_names = json.load(f)

            self.num_classes = len(self.class_names)

            # Create a fresh ResNet-50 (weights=None means no ImageNet download),
            # set up the last layer for the classes, then load the saved weights
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
            self.model.load_state_dict(
                torch.load(load_dir / "model.pth", map_location=self.device)
            )
            self.model = self.model.to(self.device)

        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Required model files not found in '{load_dir}'."
            ) from exc
        except OSError as exc:
            logger.exception("Failed to load classifier from '%s'.", load_dir)
            raise OSError(f"Could not load classifier from '{load_dir}'.") from exc

        self._trained = True
        print(f"[Classifier] Model loaded from {load_dir}")
