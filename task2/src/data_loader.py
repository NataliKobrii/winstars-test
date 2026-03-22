"""Load the animal image classification dataset.

Handles downloading from Kaggle, loading images by class folder,
and splitting into train/val/test sets.
"""

from pathlib import Path

from sklearn.model_selection import train_test_split

from config import ANIMAL_CLASSES, RANDOM_STATE
from src.logger import setup_logging

logger = setup_logging(__name__)


def download_dataset() -> Path:
    """Download the animal image dataset via kagglehub.

    Returns:
        Path:
            Path to the downloaded dataset root directory.

    Raises:
        ImportError:
            If kagglehub is not installed.
        RuntimeError:
            If dataset download fails.
    """
    # kagglehub is imported here so the rest of the code
    # still works even if kagglehub isn't installed and data_dir is provided manually
    try:
        import kagglehub
    except ImportError as exc:
        logger.exception("Failed to import kagglehub.")
        raise ImportError(
            "kagglehub is required to download the dataset. "
            "Install it with: pip install kagglehub"
        ) from exc

    try:
        path = kagglehub.dataset_download(
            "utkarshsaxenadn/animal-image-classification-dataset"
        )
    except Exception as exc:
        logger.exception("Failed to download dataset.")
        raise RuntimeError(f"Dataset download failed: {exc}") from exc

    return Path(path)


def _find_class_root(data_dir: Path) -> Path:
    """Find the directory that contains the class folders.

    The Kaggle dataset nests things like:
        data_dir/Training Data/Training Data/Cat/...
    So we can't just look directly in data_dir. Instead we walk down
    recursively until we find a folder matching one of our class names.

    Parameters:
        data_dir: Path
            Root directory to search from.

    Returns:
        Path:
            The parent directory that contains the class folders.
    """
    # Check if class folders are directly in data_dir (e.g. data_dir/cat/)
    reference_class = ANIMAL_CLASSES[0]
    if (data_dir / reference_class).is_dir() or (
        data_dir / reference_class.capitalize()
    ).is_dir():
        return data_dir

    # Otherwise search subdirectories recursively, preferring "Training Data"
    # since it has the most images
    best = None
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir() and path.name.lower() == reference_class:
            candidate = path.parent
            if "training" in str(candidate).lower():
                return candidate
            if best is None:
                best = candidate

    return best or data_dir


def load_animal_dataset(
    data_dir: Path | None = None,
) -> tuple[list[Path], list[str]]:
    """Load image paths and labels from the dataset directory.

    Expects a directory structure where each subdirectory name is a
    class label containing the corresponding images.

    Parameters:
        data_dir: Path | None
            Root directory of the dataset. If None, downloads automatically.

    Returns:
        tuple[list[Path], list[str]]:
            List of image file paths and corresponding class labels.

    Raises:
        FileNotFoundError:
            If the dataset directory does not exist.
        RuntimeError:
            If no images are found.
    """
    # If no directory given, download from Kaggle automatically
    if data_dir is None:
        data_dir = download_dataset()

    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    image_paths: list[Path] = []
    labels: list[str] = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # The Kaggle dataset has the following nested structure:
    #   versions/4/Training Data/Training Data/Cat/...
    # So the class folders aren't directly under data_dir.
    # The code searches recursively for the first folder matching a known class name.
    search_dir = _find_class_root(data_dir)

    # Go through each class folder (e.g. Cat/, Dog/ etc).
    for class_name in ANIMAL_CLASSES:
        # The dataset uses capitalized folder names ("Cat" not "cat"),
        # so both lowercase and capitalized are checked
        class_dir = search_dir / class_name
        if not class_dir.exists():
            class_dir = search_dir / class_name.capitalize()
        if not class_dir.exists():
            logger.warning("Class directory not found: %s", class_name)
            continue

        # Collect all valid image files and assign the folder name as label
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(img_file)
                labels.append(class_name)

    if not image_paths:
        raise RuntimeError(f"No images found in {data_dir}")

    print(f"Loaded {len(image_paths)} images across {len(set(labels))} classes")
    return image_paths, labels


def get_train_val_test_split(
    image_paths: list[Path],
    labels: list[str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[list[Path], list[str], list[Path], list[str], list[Path], list[str]]:
    """Split the dataset into train, validation, and test sets.

    Parameters:
        image_paths: list[Path]
            All image file paths.
        labels: list[str]
            Corresponding class labels.
        val_ratio: float
            Fraction of data for validation.
        test_ratio: float
            Fraction of data for testing.

    Returns:
        tuple:
            train_paths, train_labels, val_paths, val_labels,
            test_paths, test_labels.

    Raises:
        ValueError:
            If inputs are empty or ratios are invalid.
    """
    if not image_paths or not labels:
        raise ValueError("image_paths and labels must not be empty.")

    if not (0 < val_ratio < 1) or not (0 < test_ratio < 1):
        raise ValueError("val_ratio and test_ratio must be in (0, 1).")

    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    # Two splits are needed because sklearn's train_test_split only splits into 2 parts.
    # First split: separate out the test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_ratio,
        random_state=RANDOM_STATE,
        stratify=labels,  # keep class proportions balanced in each split
    )

    # Second split: now split the remaining data into training and validation.
    # The test set was already taken out, so 85% of the data is left.
    # To get 15% validation from the original, 15/85 of what's left is needed.
    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=adjusted_val_ratio,
        random_state=RANDOM_STATE,
        stratify=train_val_labels,
    )

    print(
        f"Split: {len(train_paths)} train | "
        f"{len(val_paths)} val | "
        f"{len(test_paths)} test"
    )

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels
