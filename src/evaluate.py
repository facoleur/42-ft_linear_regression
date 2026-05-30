import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt


def sst(real: pd.Series) -> float:
    return sum((real - np.average(real)) ** 2)


def ssr(real: pd.Series, pred: pd.Series) -> float:
    return sum((real - pred) ** 2)


def sse(real: pd.Series, pred: pd.Series) -> float:
    return sum((pred - np.average(real)) ** 2)


def r2(real: pd.Series, pred: pd.Series):
    return 1 - ssr(real, pred) / sst(real)


def mae(real: pd.Series, pred: pd.Series) -> float:
    return (1 / len(real)) * sum(abs(real - pred))


def mse(real: pd.Series, pred: pd.Series) -> float:
    return (1 / len(real)) * sum((real - pred) ** 2)


def rmse(real: pd.Series, pred: pd.Series) -> float:
    return sqrt(mse(real, pred))


def mape(real: pd.Series, pred: pd.Series) -> float:
    return 1 / len(real) * sum(abs((real - pred) / real))


def metrics(real: pd.Series, pred: pd.Series) -> pd.Series:
    """
    Evaluate the precision of a model based on 5 metrics:
    r2, mae, mse, rmse, mape.
    returns a series containing those values for a real and prediction series.
    """
    return pd.Series(
        [
            r2(real, pred),
            mae(real, pred),
            mse(real, pred),
            rmse(real, pred),
            mape(real, pred),
        ],
        index=["r2", "mae", "mse", "rmse", "mape"],
    )


def evaluate(
    train_real: pd.Series,
    train_pred: pd.Series,
    test_real: pd.Series,
    test_pred: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate the precision of a model.
    Returns a DataFrame with train, test columns to compare
    """

    # plt.scatter(test_real, test_pred)
    # plt.plot(
    #     [min(test_real), max(test_real)],
    #     [min(test_real), max(test_real)],
    # )
    # plt.savefig("eval.png")

    return pd.DataFrame(
        {
            "train": metrics(train_real, train_pred),
            "test": metrics(test_real, test_pred),
        },
    )
