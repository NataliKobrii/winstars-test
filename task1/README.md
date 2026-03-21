# Task 1. Image Classification + OOP

## Overview
This task implements three different classification models for the MNIST dataset:
1) Random Forest;
2) Feed-Forward Neural Network;
3) Convolutional Neural Network.

Each model implements a common abstract interface (MnistClassifierInterface) with two abstract methods — train and predict.
The models are accessed through a wrapper (MnistClassifier), which selects the algorithm based on the provided parameter.

## Used Machine Learning Libraries

No single framework covers both a classical ensemble model and deep neural
networks, so I used **scikit-learn** for the Random Forest and **PyTorch**
for the two neural networks.

- **Random Forest — scikit-learn**
  scikit-learn is the go-to library for classical ML in Python. It has a
  ready-made `RandomForestClassifier` [1] that trains an ensemble of
  decision trees on random subsets of the data and averages their votes to
  reduce overfitting [2]. I went with scikit-learn here simply because
  PyTorch doesn't have a built-in Random Forest, and writing one from
  scratch would be reinventing the wheel.

- **Feed-Forward NN and CNN — PyTorch**
  For the neural networks I preferred PyTorch over Keras because it makes the training loop
  explicit: you zero the gradients, do the forward pass, call
  `loss.backward()`, and step the optimizer yourself. With Keras you'd call
  `model.fit()` and the loop stays hidden, which is convenient but doesn't
  really show what's going on under the hood.

**References**:

1. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
2. https://scikit-learn.org/stable/modules/ensemble.html
3. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

## Project Structure

  **Entry point**: `main.py` runs all models
  **Configuration**: `config.py` stores hyperparameters
  **Notebook**: `demo.ipynb` demonstrates usage and results

  **Core logic** (`src/`):
    `interface.py` — defines the common classifier interface (MnistClassifierInterface)
    `classifier.py` — wrapper that selects the model by algorithm (MnistClassifier)
    `data_loader.py` — handles MNIST dataset loading
    `logger.py` — logging configuration
    `utils.py` — evaluation metrics and plotting helpers
    `validators.py` — shared input validation
    `models/` — implementations of RF, NN, and CNN

  **Artifacts**:
    `data/` — dataset (auto-downloaded)
    `outputs/` — generated figures
    `logs/` — log files

## Setup
```bash
# 1. Clone the repository and enter the project folder
git clone https://github.com/NataliKobrii/winstars-test.git
cd task1

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
python -m ipykernel install --user --name=task1 --display-name="Task 1 (venv)"
```

## Development

For linting and code quality, the project uses [Ruff](https://docs.astral.sh/ruff/).
```bash
pip install -r requirements-dev.txt
```

## Run
```bash
# Train and evaluate all three models
python main.py

# Open the demo notebook
jupyter notebook demo.ipynb
```

## Edge Cases

The implementation handles several edge cases to ensure consistent and reliable behavior:
  Calling `predict()` before `train()` raises a `RuntimeError` to prevent using an uninitialized model
  Invalid algorithm name raises a `ValueError` with a clear message
  Input with incorrect shape (e.g. `(32, 32)`) raises a `ValueError`
  2D input without batch dimension (e.g. `(28, 28)` instead of `(1, 28, 28)`) is caught
  Wrong input type (e.g. numpy instead of torch.Tensor) raises a `TypeError`
  Single image input `(1, 28, 28)` works correctly and returns an array of shape `(1,)`
  Empty input array raises an error instead of failing silently
  Blank or all-white image returns a prediction without crashing
  Invalid hyperparameters (e.g. `epochs=0`, `lr=5.0`) are rejected at model creation time

## Code Style Notes

- **Docstrings** follow the NumPy/Google style with Parameters/Returns/Raises sections,
  as taught in Python courses at York University.
- **Exception handling** and input validation patterns come from the SoftServe Academy
  Python course — validating inputs early and raising specific exceptions with clear messages.

## Requirements

  Python 3.10 - 3.12
  See `requirements.txt` for runtime dependencies
  See `requirements-dev.txt` for development tools
  