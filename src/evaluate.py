import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt


def sst(real: pd.Series) -> float:
    """Return the total sum of squares for the real target values."""
    return sum((real - np.average(real)) ** 2)


def ssr(real: pd.Series, pred: pd.Series) -> float:
    """Return the residual sum of squares between real and predicted values."""
    return sum((real - pred) ** 2)


def sse(real: pd.Series, pred: pd.Series) -> float:
    """Return the explained sum of squares for the predicted values."""
    return sum((pred - np.average(real)) ** 2)


def r2(real: pd.Series, pred: pd.Series):
    """Return the coefficient of determination for real and predicted values."""
    return 1 - ssr(real, pred) / sst(real)


def mae(real: pd.Series, pred: pd.Series) -> float:
    """Return the mean absolute error between real and predicted values."""
    return (1 / len(real)) * sum(abs(real - pred))


def mse(real: pd.Series, pred: pd.Series) -> float:
    """Return the mean squared error between real and predicted values."""
    return (1 / len(real)) * sum((real - pred) ** 2)


def rmse(real: pd.Series, pred: pd.Series) -> float:
    """Return the root mean squared error between real and predicted values."""
    return sqrt(mse(real, pred))


def mape(real: pd.Series, pred: pd.Series) -> float:
    """Return the mean absolute percentage error between real and predicted values."""
    return 1 / len(real) * sum(abs((real - pred) / real))


def metrics(real: pd.Series, pred: pd.Series) -> pd.Series:
    """Return common regression metrics for real and predicted values."""
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
    """Return train and test regression metrics as a comparison DataFrame."""

    return pd.DataFrame(
        {
            "train": metrics(train_real, train_pred),
            "test": metrics(test_real, test_pred),
        },
    )
