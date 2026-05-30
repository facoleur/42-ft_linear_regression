import matplotlib.pyplot as plt
import pandas as pd


def plot_scatter(real, pred, ax, title, xlabel=None, ylabel=None):
    ax.scatter(real, pred)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_line(x, y, ax):
    ax.plot(x, y)


# class Plot:
#     class Evaluate:
#         def compare(self, real: pd.Series, pred: pd.Series):
#             plt.scatter(real, pred)
#             plt.plot(
#                 [min(real), max(real)],
#                 [min(real), max(real)],
#             )
#             plt.savefig("compare.png")
