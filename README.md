# Healthcare Costs Predictor (Regression)

A neural network regression model built with **TensorFlow and Keras** that predicts a person's healthcare expenses from their personal data. Completed as part of the freeCodeCamp *Machine Learning with Python* certification.

**Goal:** Mean Absolute Error (MAE) under 3500 on unseen test data
**Result:** `[your final MAE]`

## Dataset

Each row describes one person:

| Feature | Type |
|---|---|
| age | numeric |
| sex | categorical |
| bmi | numeric |
| children | numeric |
| smoker | categorical |
| region | categorical |
| **expenses** | **target (healthcare cost)** |

## Approach

**1. Preprocessing**
- Categorical columns (`sex`, `smoker`, `region`) are converted to numbers with one-hot encoding.
- The data is split **80% training / 20% test**.
- The `expenses` column is popped off each set to create `train_labels` and `test_labels`.

**2. Model** – a `Sequential` network:

```
Normalization
Dense(64, ReLU)
Dense(64, ReLU)
Dense(1)
```

The normalization layer is part of the model, so raw data can be passed in at prediction time.

**3. Training**
- Optimizer: Adam (learning rate 0.001)
- Loss: mean absolute error
- Up to 300 epochs, 20% of the training data held out for validation
- Early stopping restores the best weights to avoid overfitting

**4. Evaluation** – `model.evaluate` on the unseen test set must return an MAE under 3500, meaning predictions are off by less than $3,500 on average. The final cell also graphs true vs. predicted expenses.

## Results

| Metric | Value |
|---|---|
| Test MAE | `[xxxx]` |
| Test MSE | `[xxxx]` |

## Tech Stack

Python 3 · TensorFlow / Keras · pandas · NumPy · Matplotlib · Google Colaboratory

## How to Run

1. Open the notebook in Google Colab and save a copy.
2. Run the cells in order (data downloads in cell 2).
3. Run the preprocessing, model and training cells.
4. Run the final cell to check the MAE and see the prediction graph.
5. Turn on link sharing ("anyone with the link") before submitting.

## Possible Improvements

- Tune layer sizes, learning rate and batch size
- Try other models (random forest, gradient boosting) for comparison
- Add engineered features, such as a BMI category or a smoker × BMI interaction
