import numpy as np
import pandas as pd


def train_test_split(df: pd.DataFrame, ratio=0.8, seed=42):
    """Randomly split a DataFrame into train and test sets."""
    train = df.sample(frac=ratio, random_state=seed)
    test = df.drop(train.index)
    return train, test
