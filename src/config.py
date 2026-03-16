from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 0

VAL_SPLIT = 0.2
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
