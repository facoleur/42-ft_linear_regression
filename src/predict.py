import pandas as pd
from shared import Scaler
from shared import load_model


def estimate_price(t1, t2, mileage):
    return t1 + (t2 * mileage)


def predict(km):
    """
    predict price for a given km based on thetas. Thetas are coming from the model.csv file.
    """
    t0, t1 = load_model()
    estimated_price = estimate_price(t0, t1, km)
    return estimated_price


def main():
    mileage = input("Enter car mileage: ")
    p = predict(float(mileage))
    print("estimated price is: ", p)


if __name__ == "__main__":
    main()
