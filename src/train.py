from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import DATASET_PATH, EPOCH, LEARNING_RATE, MODEL_PATH
from shared import load_model
from predict import estimate_price
from predict import predict
from shared import Scaler
from split import train_test_split


def train(X: pd.Series, y: pd.Series) -> tuple[float, float, list]:
    """Train a linear regression model and return its parameters and cost history."""

    t0, t1 = 0.0, 0.0

    km_scaler = Scaler(X.min(), X.max())
    price_scaler = Scaler(y.min(), y.max())

    X = pd.Series(km_scaler.normalize(X))
    y = pd.Series(price_scaler.normalize(y))

    costs = []
    for _ in range(0, EPOCH):
        y_pred = estimate_price(t0, t1, X)
        losses = y_pred - y

        losses = pd.Series(losses)
        t0_signal = np.average(losses)
        t1_signal = np.average(pd.Series(losses) * X)
        t0 -= t0_signal * LEARNING_RATE
        t1 -= t1_signal * LEARNING_RATE

        cost = np.average(losses**2)
        costs.append(cost)

    t1_real = t1 * (price_scaler.range / km_scaler.range)
    t0_real = (
        t0 * price_scaler.range + price_scaler.min - t1_real * km_scaler.min
    )

    return t0_real, t1_real, costs


def main():
    """Train the model from the dataset and save learned parameters to CSV."""
    df = pd.read_csv(DATASET_PATH)
    start_time = datetime.now().timestamp() * 1000
    train_df, _ = train_test_split(df)
    t0, t1, _ = train(train_df["km"], train_df["price"])
    thetas = pd.DataFrame({"t0": [t0], "t1": [t1]})
    end_time = datetime.now().timestamp() * 1000

    print(
        f"successfully trained model on {len(train_df)} datapoints in {int(end_time - start_time)} ms"
    )
    thetas.to_csv(MODEL_PATH, index=False)


if __name__ == "__main__":
    main()
