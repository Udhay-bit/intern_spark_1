# Iris Classification Model

## Project Summary

This project classifies Iris flowers into one of three species:

- setosa
- versicolor
- virginica

The final selected model is a **K-Nearest Neighbors (KNN)** classifier with:

- `n_neighbors = 5`
- StandardScaler preprocessing
- Train/test split: 80/20
- Random state: 457

## Saved Model

File:

`iris_knn_model.joblib`

The saved file contains:

- the complete inference pipeline
  - `StandardScaler`
  - `KNeighborsClassifier`
- target class names
- feature names
- test accuracy metadata

The scaler is included inside the pipeline. This is important because inference data must be transformed using the same preprocessing logic used during training.

## Installation

```bash
pip install scikit-learn joblib numpy
```

## Inference Example

```python
import joblib
import numpy as np

bundle = joblib.load("iris_knn_model.joblib")

model = bundle["model"]
target_names = bundle["target_names"]

# Feature order:
# [sepal length, sepal width, petal length, petal width]

sample = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

prediction = model.predict(sample)

predicted_class = target_names[prediction[0]]

print("Predicted class:", predicted_class)

# Optional: prediction probabilities
probabilities = model.predict_proba(sample)
print("Probabilities:", probabilities)
```

## Expected Input Format

The model expects four numerical features in this exact order:

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

Example:

```python
[[5.1, 3.5, 1.4, 0.2]]
```

## Output

The model returns a numeric class:

- `0` → setosa
- `1` → versicolor
- `2` → virginica

Use `target_names` from the saved bundle to convert the numeric prediction into the species name.

## Reproducibility

The model was trained using the Iris dataset with:

```text
train_test_split(..., test_size=0.2, random_state=457)
KNeighborsClassifier(n_neighbors=5)
```

Test accuracy from this reproduced training run: 0.9667
