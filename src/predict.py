from shared import load_model


def estimate_price(t1, t2, mileage):
    """Return the estimated price for a mileage
    using linear model parameters."""
    return t1 + (t2 * mileage)


def predict(km):
    """Predict the car price for a mileage using the saved model parameters."""
    t0, t1 = load_model()
    estimated_price = estimate_price(t0, t1, km)
    return estimated_price


def main():
    """Prompt for a mileage value and print the predicted car price."""
    mileage = input("Enter car mileage: ")
    p = predict(float(mileage))
    print("estimated price is: ", p)


if __name__ == "__main__":
    main()
