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


def train(data: pd.DataFrame) -> tuple[float, float, list]:
    """
    train a model on a dataset with km, price columns to determine price based on mileage.
    define t0,t1
    """
    # raw_data = data.copy()

    t0, t1 = 0.0, 0.0
    km = pd.Series(data["km"])
    price = pd.Series(data["price"])

    km_scaler = Scaler(km.min(), km.max())
    price_scaler = Scaler(price.min(), price.max())

    km = pd.Series(km_scaler.normalize(km))
    price = pd.Series(price_scaler.normalize(price))

    costs = []
    for _ in range(0, EPOCH):
        estimated_price = estimate_price(t0, t1, km)
        losses = estimated_price - price

        losses = pd.Series(losses)
        t0_signal = np.average(losses)
        t1_signal = np.average(pd.Series(losses) * km)
        t0 -= t0_signal * LEARNING_RATE
        t1 -= t1_signal * LEARNING_RATE

        cost = np.average(losses**2)
        costs.append(cost)

    t1_real = t1 * (price_scaler.range / km_scaler.range)
    t0_real = (
        t0 * price_scaler.range + price_scaler.min - t1_real * km_scaler.min
    )

    thetas = pd.DataFrame(
        data={
            "t0": [t0_real],
            "t1": [t1_real],
        }
    )

    return t0_real, t1_real, costs

    # x = np.linspace(km.min(), km.max())
    # y = t0 + t1 * x
    # x_raw = km_scaler.denormalize(x)
    # y_raw = price_scaler.denormalize(y)

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    # ax1.scatter(raw_data["km"], raw_data["price"])
    # ax1.plot(x_raw, y_raw)
    # ax1.set_title("price/km regression")

    # ax2.plot(costs)
    # ax2.set_title("cost reduction")
    # fig.savefig("viz.png")

    return thetas


def main():
    """
    export model in DATASET_PATH.
    model contains: t0,t1 and values used by the Scaler to normalize data.
    """
    df = pd.read_csv(DATASET_PATH)
    start_time = datetime.now().timestamp() * 1000
    train_df, test_df = train_test_split(df)
    thetas = train(train_df)
    end_time = datetime.now().timestamp() * 1000

    print(
        f"successfully trained model on {len(train_df)} datapoints in {int(end_time - start_time)} ms"
    )
    thetas.to_csv(MODEL_PATH, index=False)


if __name__ == "__main__":
    main()
