from dataclasses import dataclass
import pandas as pd
import sys

from config import MODEL_PATH


@dataclass
class Scaler:
    """Normalize and denormalize values using a fixed minimum and maximum."""

    min: float
    max: float

    def normalize(self, x: pd.Series | float):
        """Scale a value or Series to the 0-1 range."""
        return (x - self.min) / (self.max - self.min)

    def denormalize(self, x: pd.Series | float):
        """Convert a normalized value or Series back to the original range."""
        return x * (self.max - self.min) + self.min

    @property
    def range(self):
        """Return the distance between the configured maximum and minimum."""
        return self.max - self.min


def load_model():
    """Load and return the saved linear model parameters from MODEL_PATH."""
    try:
        t = pd.read_csv(MODEL_PATH)
        return (t["t0"].iloc[0], t["t1"].iloc[0])
    except:
        print("Model is not trained. run `train.py` first")
        sys.exit(1)
