"""Project configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Paths
MODEL_SAVE_DIR = BASE_DIR / "outputs" / "models"
NER_MODEL_DIR = MODEL_SAVE_DIR / "ner"
CLASSIFIER_MODEL_DIR = MODEL_SAVE_DIR / "classifier"
FIGURE_SAVE_DIR = BASE_DIR / "outputs" / "figures"

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "main.log"
LOG_LEVEL = "INFO"

# Animal classes (15 classes from Kaggle dataset)
ANIMAL_CLASSES = [
    "beetle",
    "butterfly",
    "cat",
    "cow",
    "dog",
    "elephant",
    "gorilla",
    "hippo",
    "lizard",
    "monkey",
    "mouse",
    "panda",
    "spider",
    "tiger",
    "zebra",
]

# NER hyperparameters
# Values follow HuggingFace fine-tuning recommendations:
# https://huggingface.co/docs/transformers/training#finetune-a-pretrained-model
# and the original BERT paper (Devlin et al., 2019), Appendix A.3:
# https://arxiv.org/abs/1810.04805
NER_MODEL_NAME = "bert-base-uncased"  # 110M params, uncased works fine for entity names
NER_EPOCHS = 3  # paper recommends 2-4; picked 3 as a middle ground
NER_BATCH_SIZE = 16  # paper recommends 16 or 32; picked 16 to use less memory on CPU
NER_LR = 5e-5  # paper recommends {2e-5, 3e-5, 5e-5}; picked 5e-5 since the dataset is small and synthetic
NER_MAX_LENGTH = 64  # the sentences are short, no need for BERT's full 512

# NER label mapping using BIO tagging scheme
# (standard for token classification, see HuggingFace NER guide:
#  https://huggingface.co/docs/transformers/tasks/token_classification)
# B- indicates the beginning of an entity.
# I- indicates a token is contained inside the same entity (for example, the State token is a part of an entity like Empire State Building).
# 0 indicates the token doesn’t correspond to any entity.
# LABEL2ID: during training, "B-ANIMAL" is converted to 1 because the loss function (CrossEntropyLoss) needs integer targets
# ID2LABEL: during inference, the model outputs 1, which is converted back to "B-ANIMAL" for readability.
# (https://huggingface.co/docs/transformers/tasks/token_classification#preprocess)
NER_LABEL2ID = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
NER_ID2LABEL = {v: k for k, v in NER_LABEL2ID.items()}

# Classifier hyperparameters
# See the PyTorch transfer learning tutorial for guidance:
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
CLF_EPOCHS = 10  # tried 5 first but accuracy was still improving, 10 gave ~89%
CLF_BATCH_SIZE = (
    32  # 32 is what the tutorial uses as it is a standard for image classification
)
CLF_LR = 0.001  # default Adam lr from PyTorch docs (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
# ResNet comes in 18/34/50/101/152 variants. Picked 50 as a balance (https://pytorch.org/vision/stable/models/resnet.html).
# 224 is the input size the ImageNet pretrained weights were trained on,
# so images must be resized to match (https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html).
CLF_IMAGE_SIZE = 224

# General
RANDOM_STATE = 42  # fixed seed so results are reproducible

# Development
DEBUG = False
