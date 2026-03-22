# Task 2. Named Entity Recognition + Image Classification

## Overview
This task builds an ML pipeline that determines whether an image contains the animal mentioned in a given text. It combines two models:
1) BERT-based NER model — extracts animal names from text;
2) ResNet-50 image classifier — identifies the animal in the image.

The pipeline takes a text string and an image as input and returns a single boolean value: `True` if the text matches the image, `False` otherwise.

## Used Machine Learning Libraries

The task requires both an NLP model and a Computer Vision model, so I used
**HuggingFace Transformers** for the NER and **PyTorch + torchvision** for the
image classifier.

- **NER — HuggingFace Transformers (BERT)**
  The task requires a transformer-based model (not LLM). I chose
  `bert-base-uncased` [1] because it's a well-established pretrained
  language model that works well for token classification tasks like NER.
  I fine-tuned it with BIO tagging (B-ANIMAL, I-ANIMAL, O) to extract
  animal names from text. Since there's no existing NER dataset specifically
  for animal names, I generated synthetic training data from sentence
  templates.

- **Image Classifier — PyTorch (ResNet-50)**
  For the image classifier I went with ResNet-50 [2] because it already
  knows how to recognize visual features like edges, textures, and shapes
  from being trained on ImageNet. Instead of training from scratch, 
  I just swapped out the last layer so it outputs my 15
  animal classes instead of the original 1000 ImageNet categories.

- **Dataset — Kaggle Animal Image Classification**
  I used a publicly available dataset [3] with 15 animal classes and
  around 2000 images per class (30,000 total). The dataset is downloaded
  automatically via `kagglehub`.

**References**:

1. https://huggingface.co/docs/transformers/model_doc/bert
2. https://pytorch.org/vision/stable/models/resnet.html
3. https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset

## Project Structure

  **Entry point**: `main.py` runs the full pipeline (text + image → boolean)
  **Configuration**: `config.py` stores hyperparameters and paths
  **Notebook**: `demo.ipynb` demonstrates EDA, training, and pipeline usage

  **Core logic** (`src/`):
    `pipeline.py` — AnimalVerificationPipeline (combines both models)
    `data_loader.py` — dataset download, loading, and splitting
    `ner_data_generator.py` — synthetic BIO-tagged sentence generation
    `logger.py` — logging configuration
    `utils.py` — evaluation metrics and plotting helpers
    `validators.py` — shared input validation
    `models/` — AnimalNERModel (BERT) and AnimalImageClassifier (ResNet-50)

  **Artifacts**:
    `outputs/` — saved models and generated figures
    `logs/` — log files

## Setup
```bash
# 1. Clone the repository and enter the project folder
git clone https://github.com/NataliKobrii/winstars-test.git
cd task2

# 2. Create a virtual environment to isolate dependencies
#    and prevent conflicts with other projects
python3 -m venv venv

# Activate — macOS / Linux
source venv/bin/activate
# Activate — Windows
venv\Scripts\activate

# 3. Install dependencies
# Python 3.10–3.12 is recommended (PyTorch does not yet support 3.13+)
pip install -r requirements.txt

# If torch fails to install (common on older macOS, Linux ARM, or Python 3.13+),
# install it separately using the official command for your platform:
#   https://pytorch.org/get-started/locally/
#
# CPU-only example (works on any platform):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#
# Then re-run to install the remaining dependencies:
#   pip install -r requirements.txt

# 4. Register the venv as a Jupyter kernel (required for demo.ipynb)
pip install ipykernel
python -m ipykernel install --user --name=task2 --display-name="Task 2 (venv)"
```

## Development

For linting and code quality, the project uses [Ruff](https://docs.astral.sh/ruff/).
```bash
pip install -r requirements-dev.txt
```

## Run
```bash
# 1. Train both models (one-time)
python train_ner.py
python train_classifier.py

# 2. Run the full pipeline (use any image from the Kaggle dataset cache)
#    After training, images are cached at:
#    ~/.cache/kagglehub/datasets/utkarshsaxenadn/animal-image-classification-dataset/versions/4/Testing Data/Testing Data/<ClassName>/
python main.py --text "There is a cow in the picture" --image "~/.cache/kagglehub/datasets/utkarshsaxenadn/animal-image-classification-dataset/versions/4/Testing Data/Testing Data/Cow/Cow-Test (1).jpeg"
# Output: True

python main.py --text "There is a dog in the picture" --image "~/.cache/kagglehub/datasets/utkarshsaxenadn/animal-image-classification-dataset/versions/4/Testing Data/Testing Data/Cow/Cow-Test (1).jpeg"
# Output: False

# 3. Run individual models separately
python infer_ner.py --text "There is a cow in the picture"
# Output: ['cow']

python infer_classifier.py --image "~/.cache/kagglehub/datasets/utkarshsaxenadn/animal-image-classification-dataset/versions/4/Testing Data/Testing Data/Cow/Cow-Test (1).jpeg"
# Output: cow

# 4. Open the demo notebook
jupyter notebook demo.ipynb
```

### Training with custom parameters
```bash
python train_ner.py --epochs 5 --batch-size 32 --lr 3e-5
python train_classifier.py --epochs 15 --batch-size 64 --lr 0.0005
```

## Demo Notebook

The `demo.ipynb` notebook contains:
1. **EDA**: Dataset statistics, sample images, class distribution, image size analysis
2. **NER Demo**: Synthetic data generation, model training, entity extraction examples
3. **Classifier Demo**: Model training, test set evaluation, confusion matrix, sample predictions
4. **Pipeline Demo**: End-to-end verification with positive, negative, and edge case examples

To run:
```bash
jupyter notebook demo.ipynb
```
Run all cells top to bottom. The notebook trains both models and runs the full pipeline demo.

## Edge Cases

The implementation handles several edge cases to ensure consistent and reliable behavior:
  Calling `predict()` before `train()` or `load()` raises a `RuntimeError`
  Empty or blank text input raises a `ValueError`
  Missing image file raises a `FileNotFoundError`
  Text with no animal mentioned returns `False` (NER extracts empty list)
  Case-insensitive matching — "Cat" in text matches "cat" from classifier
  Invalid hyperparameters (e.g. `epochs=0`, `lr=5.0`) are rejected at model creation time

## Known Limitations

- **NER recognizes only exact animal names** from the training set (beetle, butterfly, cat, cow, dog, elephant, gorilla, hippo, lizard, monkey, mouse, panda, spider, tiger, zebra). Synonyms, slang, or diminutives (e.g. "doggy", "kitty", "puppy") will be extracted as-is but won't match the classifier's class names, leading to a `False` result even if the image is correct.
- **The classifier is limited to 15 classes.** Animals outside this set will be misclassified as the closest known class.
- **NER is trained on synthetic data**, so unusual sentence structures may not be handled perfectly.

## Code Style Notes

- **Docstrings** follow the NumPy/Google style with Parameters/Returns/Raises sections,
  as taught in Python courses at York University.
- **Exception handling** and input validation patterns come from the SoftServe Academy
  Python course — validating inputs early and raising specific exceptions with clear messages.

## Requirements

  Python 3.10 - 3.12
  See `requirements.txt` for runtime dependencies
  See `requirements-dev.txt` for development tools
  