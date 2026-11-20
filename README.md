# Handwritten Digit Classification with MLPClassifier

A Python notebook demonstrating how to load, inspect, visualize, preprocess, and classify handwritten digits using a Multi-Layer Perceptron (`MLPClassifier`) from `scikit-learn` on the built-in `digits` dataset.

---

## Overview

This notebook walks through a complete end-to-end machine learning workflow on small 8x8 grayscale digit images:

1. **Dataset Loading & Exploration**: Loads `sklearn.datasets.load_digits()` and inspects available attributes and raw pixel matrices.
2. **Visualization**: Renders a 4x4 grid of sample digits with their ground-truth labels.
3. **Preprocessing & Train/Test Split**: Flattens 8x8 image matrices into 64-feature vectors and splits the 1,797 samples into 1,000 training samples and 797 test samples.
4. **Model Training**: Trains an `MLPClassifier` with a single hidden layer of 15 neurons, logistic activation, and stochastic gradient descent (`sgd`), while plotting the training loss curve.
5. **Evaluation & Error Analysis**: Computes test set accuracy (~91.47%) and visualizes misclassified examples alongside predicted and actual labels.

---

## Key Details from the Notebook

| Component | Details |
| :--- | :--- |
| **Dataset** | `sklearn.datasets.load_digits()` (1,797 samples, 8x8 pixel images, 10 digit classes `0`–`9`) |
| **Feature Shape** | Flattened from `(1797, 8, 8)` to `(1797, 64)` |
| **Train / Test Split** | `x[:1000]`, `y[:1000]` for training; `x[1000:]`, `y[1000:]` (797 samples) for testing |
| **Model** | `MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1, verbose=True)` |
| **Training Behavior** | Ran for 189 iterations before early stopping due to loss not improving by more than `tol=0.0001` for 10 consecutive epochs |
| **Test Accuracy** | `0.9146800501882058` (~91.47%, with 68 misclassified samples out of 797) |

---

## Prerequisites

* Python 3
* `scikit-learn`
* `matplotlib`
* `numpy` (installed alongside `scikit-learn` / `matplotlib`)

Install the required packages via `pip`:

```bash
pip install scikit-learn matplotlib
