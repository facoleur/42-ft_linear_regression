import pandas as pd

from config import RANDOM_STATE


def train_test_split(df: pd.DataFrame, ratio=0.8, seed=RANDOM_STATE):
    """Randomly split a DataFrame into train and test sets."""
    train = df.sample(frac=ratio, random_state=seed)
    test = df.drop(train.index)
    return train, test
