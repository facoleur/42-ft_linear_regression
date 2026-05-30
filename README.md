# ft_linear_regression

Linear regression from scratch (gradient descent) to predict car price from mileage.

## Constraint

No `scikit-learn` (or equivalent ML library). Only `numpy`, `pandas`, `matplotlib`.

## Usage

Run from the `src/` directory.



```bash
cd src
```

### Train

Trains the model on `data/data.csv` and writes parameters to `data/model.csv`.

```bash
python train.py
```

### Predict

Prompts for a mileage and prints the estimated price (uses `data/model.csv`).

```bash
python predict.py
```

### Main

Trains, evaluates (train/test split) and saves a summary plot.

```bash
python main.py
```

## Output

- `data/model.csv` — learned parameters `t0`, `t1` (price = t0 + t1 * km).
- `vizualisation/viz.png` — dataset + fit, residuals, residuals histogram, cost curve, real vs predicted.
