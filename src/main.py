import numpy as np
from config import DATASET_PATH
from plot import plot_line, plot_scatter
from train import train
from predict import predict
from split import train_test_split
import pandas as pd
from evaluate import sse, ssr, sst, r2, mae, mse, rmse, mape, evaluate
import sys
import matplotlib.pyplot as plt


def main():
    try:
        df = pd.read_csv(DATASET_PATH)
    except:
        sys.exit("csv not found.")

    train_real, test_real = train_test_split(df)

    X_train = train_real["km"]
    y_train = train_real["price"]
    X_test = test_real["km"]
    y_test = test_real["price"]

    t0, t1, costs = train(train_real)

    y_pred_test = predict(X_test)
    y_pred_train = predict(X_train)

    eval = evaluate(
        y_train,
        y_pred_train,
        X_test,
        y_pred_test,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.4)
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # DATA VIZ
    # AX1
    ax1.scatter(df["km"], df["price"])
    ax1.set_title("Dataset")
    ax1.set_xlabel("km")
    ax1.set_ylabel("price")

    x = np.linspace(df["km"].min(), df["km"].max())
    y = t0 + t1 * x
    ax1.plot(x, y, color="black")

    # AX2
    ax2.set_title("Residual plot")
    ax2.set_xlabel("km")
    ax2.set_ylabel("price difference")
    residuals_train = y_pred_train - y_train
    residuals_test = y_pred_test - y_test

    ax2.scatter(
        train_real["km"],
        residuals_train,
        label=f"train r2={eval.loc['r2', 'train']:.2f}",
    )
    ax2.scatter(
        X_test,
        residuals_test,
        color="orange",
        label=f"test r2={eval.loc['r2', 'test']:.2f}",
    )
    ax2.axhline(0, color="black")
    ax2.legend()

    # AX3
    ax3.set_title("Histogram of residuals")
    ax3.hist(residuals_train, bins=10, orientation="horizontal")
    ax3.hist(residuals_test, bins=10, orientation="horizontal")
    ax3.sharey(ax2)

    # AX4
    ax4.plot(costs)
    ax4.set_title("Cost reduction over epoch")
    ax4.set_xlabel("epoch")
    ax4.set_ylabel("cost")

    # AX6
    plot_scatter(test_real, y_pred_test, ax6, "Real vs predicted")

    fig.savefig(VIZUALISATION_PATH)

    print(eval)


if __name__ == "__main__":
    main()
