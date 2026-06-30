# Atividade 1 — Shallow Models: Classification & Regression

> **Course:** Aprendizagem de Máquina · PPGIA / PUC-PR · MSc 2026  
> **Instructor:** Prof. Alceu de Souza Britto Jr. (alceu@ppgia.pucpr.br)  
> **Related modules:** [Module 2 — Experimental Protocols](../modules/02-protocols.md) · [Module 3 — Shallow Techniques](../modules/03-shallow.md)

---

## Overview

This activity applies the shallow supervised learning techniques from Module 3 under the experimental protocol from Module 2. Two tasks are addressed: binary classification (Part A) and regression (Part B). Both use publicly available benchmark datasets and 5-fold cross-validation as the evaluation protocol.

The objective is not only to obtain results, but to develop the capacity to **design a rigorous comparative experiment**, interpret metric differences with statistical awareness, and reason about method selection given dataset characteristics.

---

## Part A — Classification

### Dataset: Breast Cancer Wisconsin

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Instances | 569 |
| Features | 30 continuous (derived from digitised FNA images) |
| Target | Binary: 0 = malignant, 1 = benign |
| Class distribution | 212 malignant (37.3%), 357 benign (62.7%) |

Features encode geometric and textural properties of cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each computed as mean, standard error, and worst (largest) value across the image.

### Classifiers

The following nine classifiers are compared:

| Classifier | Module section | Key hyperparameters to tune |
|---|---|---|
| Decision Tree | [03-shallow § 4](../modules/03-shallow.md#4-tree-based-methods) | Max depth, criterion |
| k-Nearest Neighbours | [03-shallow § 1](../modules/03-shallow.md#1-instance-based-methods) | $k$, distance metric |
| Naive Bayes | [03-shallow § 2](../modules/03-shallow.md#2-probabilistic-methods) | Smoothing $\alpha$ |
| SVM | [03-shallow § 3](../modules/03-shallow.md#3-geometric-methods) | $C$, kernel, $\gamma$ |
| MLP | [03-shallow § 5](../modules/03-shallow.md#mlp-as-a-shallow-baseline) | Architecture, $\eta$ |
| Random Forest | [03-shallow § 5](../modules/03-shallow.md#random-forest) | $T$, $m$, depth |
| Bagging | [03-shallow § 5](../modules/03-shallow.md#bagging-bootstrap-aggregating) | $T$, base learner |
| AdaBoost | [03-shallow § 5](../modules/03-shallow.md#adaboost) | $T$, learning rate |
| XGBoost | [03-shallow § 5](../modules/03-shallow.md#xgboost-extreme-gradient-boosting) | $\eta$, depth, $\lambda$ |

### Experimental Protocol

- **Validation:** Stratified 5-fold cross-validation (stratification preserves class ratio per fold)
- **Metrics:** Accuracy, F1 score (macro), Precision (macro), Recall (macro)
- **Comparison:** Per-fold metric values enable statistical comparison across classifiers

### Questions

**A.1** — Which classifier achieves the best mean accuracy across folds? Is the difference statistically significant relative to the second-best? Apply an appropriate statistical test (see [Module 2 §4](../modules/02-protocols.md#4-comparing-classifiers-statistically)).

**A.2** — Analyse the confusion matrices for the two best classifiers. Which class (malignant / benign) is more frequently misclassified, and what are the implications for clinical deployment? Consider the relative costs of false positives and false negatives.

**A.3** — Compare the trade-off between precision and recall across classifiers. Which methods exhibit high precision at the cost of recall, and which exhibit the inverse? What does this reveal about their decision boundaries?

**A.4** — Ensemble methods (Random Forest, Bagging, AdaBoost, XGBoost) are expected to outperform their base learners. Does the experimental evidence support this? Analyse any exceptions and hypothesise why they occur.

---

## Part B — Regression

### Dataset: Diabetes

| Property | Value |
|---|---|
| Source | scikit-learn built-in (`sklearn.datasets.load_diabetes`) |
| Instances | 442 |
| Features | 10 numeric (age, sex, BMI, blood pressure, 6 serum measurements) — pre-normalised |
| Target | Continuous: disease progression measure one year after baseline (range 25–346) |

### Regressors

The following seven regressors are compared:

| Regressor | Module section | Key hyperparameters to tune |
|---|---|---|
| Decision Tree | [03-shallow § 4](../modules/03-shallow.md#decision-tree) | Max depth, min samples leaf |
| k-Nearest Neighbours | [03-shallow § 1](../modules/03-shallow.md#k-nearest-neighbours-k-nn) | $k$, distance metric |
| SVR | [03-shallow § 3](../modules/03-shallow.md#support-vector-machine-svm) | $C$, $\varepsilon$, kernel |
| MLP | [03-shallow § 5](../modules/03-shallow.md#mlp-as-a-shallow-baseline) | Architecture, $\eta$ |
| Random Forest | [03-shallow § 5](../modules/03-shallow.md#random-forest) | $T$, $m$, depth |
| Bagging | [03-shallow § 5](../modules/03-shallow.md#bagging-bootstrap-aggregating) | $T$, base learner |
| XGBoost | [03-shallow § 5](../modules/03-shallow.md#xgboost-extreme-gradient-boosting) | $\eta$, depth, $\lambda$ |

### Experimental Protocol

- **Validation:** 5-fold cross-validation
- **Metrics:** R² (coefficient of determination), MAE (mean absolute error)
- **Note:** Because the target range spans ~320 units, report MAE both in absolute terms and as a percentage of the target range for interpretability.

### Questions

**B.1** — Which regressor achieves the highest mean R² and lowest mean MAE across folds? Are the two rankings consistent (i.e., does the best R² model also yield the lowest MAE)? If not, explain what this discrepancy reveals about the error distribution of each model.

---

## Implementation Notes

### Recommended Stack

```python
# Core
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Regressors
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_absolute_error
```

### Preprocessing Considerations

- Feature standardisation (`StandardScaler`) is **required** before k-NN, SVM/SVR, and MLP. Tree-based methods and Naive Bayes are invariant to monotonic feature scaling.
- Use `Pipeline(steps=[('scaler', StandardScaler()), ('clf', classifier)])` to prevent data leakage — the scaler must be fit only on training folds.

### Reproducibility

Set a fixed random seed for all stochastic components:

```python
SEED = 42
```

Pass `random_state=SEED` to all classifiers/regressors that accept it, and to `KFold`/`StratifiedKFold`.

---

## Results

> *To be completed upon activity submission.*

### Part A — Classification Results

| Classifier | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Decision Tree | — | — | — | — |
| k-NN | — | — | — | — |
| Naive Bayes | — | — | — | — |
| SVM | — | — | — | — |
| MLP | — | — | — | — |
| Random Forest | — | — | — | — |
| Bagging | — | — | — | — |
| AdaBoost | — | — | — | — |
| XGBoost | — | — | — | — |

### Part B — Regression Results

| Regressor | R² | MAE |
|---|---|---|
| Decision Tree | — | — |
| k-NN | — | — |
| SVR | — | — |
| MLP | — | — |
| Random Forest | — | — |
| Bagging | — | — |
| XGBoost | — | — |

---

*[← README](../README.md) · [Module 2 — Experimental Protocols](../modules/02-protocols.md) · [Module 3 — Shallow Techniques](../modules/03-shallow.md)*
