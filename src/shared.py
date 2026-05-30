from dataclasses import dataclass
import pandas as pd
import sys

from config import MODEL_PATH


@dataclass
class Scaler:
    """
    Helper class to normalize and denormalize data from min and max values in a Series.
    """

    min: float
    max: float

    def normalize(self, x: pd.Series | float):
        return (x - self.min) / (self.max - self.min)

    def denormalize(self, x: pd.Series | float):
        return x * (self.max - self.min) + self.min

    @property
    def range(self):
        return self.max - self.min


def load_model():
    """
    load the model from MODEL_PATH csv.
    header columns are: t0, t1, km_min, km_max, price_min, price_max
    """
    try:
        t = pd.read_csv(MODEL_PATH)
        return (t["t0"].iloc[0], t["t1"].iloc[0])
    except:
        print("Model is not trained. run `train.py` first")
        sys.exit(1)
