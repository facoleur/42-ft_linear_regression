from pathlib import Path

# hyperparams
LEARNING_RATE = 0.02
EPOCH = 2000

# paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_PATH = DATA_DIR / "model.csv"
DATASET_PATH = DATA_DIR / "data.csv"
VIZUALISATION_PATH = ROOT_DIR / "vizualisation/viz.png"
