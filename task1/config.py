"""Project configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
RANDOM_STATE = 42

# Paths
DATA_DIR = BASE_DIR / "data"
MODEL_SAVE_DIR = BASE_DIR / "outputs" / "models"
FIGURE_SAVE_DIR = BASE_DIR / "outputs" / "figures"

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "main.log"
LOG_LEVEL = "INFO"

# Random Forest
RF_N_ESTIMATORS = 100

# Feed-Forward Neural Network
NN_HIDDEN_1 = 256
NN_HIDDEN_2 = 128
NN_DROPOUT = 0.2

# Convolutional Neural Network
CNN_FILTERS_1 = 32
CNN_FILTERS_2 = 64
CNN_HIDDEN = 128
CNN_DROPOUT = 0.3

# Development
DEBUG = False
