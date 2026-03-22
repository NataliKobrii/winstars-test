"""Animal verification pipeline, combining NER and image classification."""

from pathlib import Path

from src.logger import setup_logging
from src.models.image_classifier import AnimalImageClassifier
from src.models.ner_model import AnimalNERModel
from src.validators import validate_image_path, validate_text

logger = setup_logging(__name__)


class AnimalVerificationPipeline:
    """Pipeline that verifies whether an image contains the animal mentioned in text.

    Combines a BERT-based NER model (to extract animal names from text)
    with a ResNet-50 image classifier (to identify the animal in an image).
    """

    def __init__(self) -> None:
        """Initialize the pipeline by loading both models.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If model directories do not exist.
            OSError:
                If model loading fails.
        """
        # NER model — extracts animal names from text
        self.ner = AnimalNERModel()
        self.ner.load()  # loads trained BERT model from outputs/models/ner/

        # Image classifier — identifies the animal in the image
        self.clf = AnimalImageClassifier()
        self.clf.load()  # loads trained ResNet-50 model from outputs/models/classifier/

    def verify(self, text: str, image_path: Path) -> bool:
        """Check if the image contains the animal mentioned in the text.

        Parameters:
            text: str
                Input text describing an animal.
            image_path: Path
                Path to the image file.

        Returns:
            bool:
                True if the predicted class matches any extracted animal name.
                False otherwise.

        Raises:
            TypeError:
                If inputs have the wrong type.
            ValueError:
                If text is empty or image format is unsupported.
            FileNotFoundError:
                If the image file does not exist.
            RuntimeError:
                If inference fails.
        """
        # Check that inputs are valid before running the models
        validate_text(text)
        image_path = Path(image_path)
        validate_image_path(image_path)

        # Extract animal names from text using NER
        animals = self.ner.predict(text)
        # Classify the image to get the predicted animal
        predicted_class = self.clf.predict(image_path)

        # Check if they match by lowercasing and removing leading/trailing whitespace
        animals_lower = [a.lower().strip() for a in animals]
        return predicted_class.lower().strip() in animals_lower
