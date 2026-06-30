# Module 3 — Shallow Techniques

> **Colour:** ![#f0883e](https://placehold.co/12x12/f0883e/f0883e.png) Orange-red  
> **Prerequisites:** [Module 1 — Foundations](01-foundations.md) · [Module 2 — Experimental Protocols](02-protocols.md)  
> **Connects to:** [Module 4 — Descriptive Models](04-descriptive.md) · [Module 5 — Deep Techniques](05-deep.md) · [Atividade 1](../activities/atividade-1.md)

---

## Overview

Shallow techniques are supervised learning methods that operate on hand-crafted feature representations — as opposed to deep methods that learn representations end-to-end. Despite the label "shallow", many of these methods (SVM, ensemble methods, gradient boosting) remain state-of-the-art on structured/tabular data and constitute the empirical baseline against which deep models must be justified.

This module covers supervised classifiers and regressors, organised by their inductive bias: geometric, probabilistic, tree-based, instance-based, and ensemble methods.

---

## 1. Instance-Based Methods

### k-Nearest Neighbours (k-NN)

A non-parametric method: the prediction for a query $\mathbf{x}$ is derived from the $k$ closest training points under a chosen distance metric.

**Classification:** majority vote among $k$ neighbours.  
**Regression:** mean (or distance-weighted mean) of $k$ neighbours' target values.

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i$$

**Key hyperparameters:** $k$ (neighbourhood size), distance metric (Euclidean, Manhattan, Minkowski).

**Inductive bias:** Smoothness — nearby points have similar labels. Fails in high dimensions (curse of dimensionality) where distances become uninformative.

**Common pitfall:** Failing to standardise features before computing distances. Large-magnitude features dominate the distance computation regardless of their predictive relevance.

---

## 2. Probabilistic Methods

### Naive Bayes

Applies Bayes' theorem under the conditional independence assumption:

$$P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^d P(x_j \mid y)$$

The "naive" assumption — that features are independent given the class — is rarely true in practice but yields surprisingly competitive performance, especially in text classification and high-dimensional sparse data.

**Variants:** Gaussian NB (continuous features), Multinomial NB (count features), Bernoulli NB (binary features).

**Key hyperparameters:** Smoothing parameter $\alpha$ (Laplace/additive smoothing) to handle zero-probability estimates.

---

## 3. Geometric Methods

### Support Vector Machine (SVM)

Finds the maximum-margin hyperplane separating classes:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \; \forall i$$

The **soft-margin** extension introduces slack variables $\xi_i \geq 0$ and a regularisation parameter $C$ controlling the bias–variance trade-off:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i$$

**Kernel trick:** Maps inputs to a higher-dimensional feature space implicitly via a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$. Common kernels: RBF (Gaussian), polynomial, sigmoid.

**SVR (regression):** Uses an $\varepsilon$-insensitive loss — errors within a tube of width $\varepsilon$ around the prediction are ignored.

**Key hyperparameters:** $C$ (regularisation), kernel type, $\gamma$ (RBF bandwidth), $\varepsilon$ (SVR tube width).

---

## 4. Tree-Based Methods

### Decision Tree

Recursively partitions the feature space using axis-aligned splits. At each node, the split is chosen to maximise an impurity reduction criterion.

#### ID3 Algorithm

ID3 (Iterative Dichotomiser 3, Quinlan 1986) is the foundational decision tree learning algorithm. It selects splits using **Information Gain** — the reduction in entropy achieved by splitting on attribute $A$:

$$\text{Entropy}(S) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

where $p_k$ is the proportion of examples in class $k$ in set $S$. Entropy is 0 when all examples belong to one class (pure node) and maximal at $\log_2 K$ when classes are uniformly distributed.

**Information Gain** of splitting $S$ on attribute $A$:

$$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \cdot \text{Entropy}(S_v)$$

where $S_v$ is the subset of $S$ for which $A = v$. ID3 greedily selects the attribute $A^* = \arg\max_A \text{Gain}(S, A)$ at each node.

**Limitations of ID3:** Biased toward attributes with many values (high cardinality); no pruning; handles only categorical attributes. **C4.5** (Quinlan 1993) extends ID3 with Gain Ratio (normalises by split information), continuous attribute handling, and pruning. **CART** uses Gini impurity instead of entropy and produces binary splits.

**Impurity criteria comparison:**

| Criterion | Formula | Algorithm |
|---|---|---|
| Entropy / Information Gain | $-\sum_k p_k \log_2 p_k$ | ID3, C4.5 |
| Gini impurity | $1 - \sum_k p_k^2$ | CART |
| Variance reduction | $\text{Var}(y) - \sum_c w_c \text{Var}(y_c)$ | CART (regression) |

**Key hyperparameters:** Maximum depth, minimum samples per leaf, impurity criterion.

**Inductive bias:** Axis-aligned decision boundaries; high variance (prone to overfitting) but zero bias on training data when fully grown.

---

## 5. Ensemble Methods

Ensemble methods combine multiple base learners to reduce variance (bagging), bias (boosting), or both (stacking).

### Bagging (Bootstrap Aggregating)

Trains $T$ base models on bootstrap samples of the training data. Predictions are aggregated by majority vote (classification) or mean (regression):

$$\hat{y} = \frac{1}{T} \sum_{t=1}^T f_t(\mathbf{x})$$

Reduces variance without increasing bias. Most effective with high-variance base learners (e.g., deep decision trees).

### Random Forest

Bagging of decision trees with an additional feature subsampling step: at each split, only a random subset of $m \leq d$ features is considered. This decorrelates the trees, reducing ensemble variance further.

**Key hyperparameters:** Number of trees $T$, features per split $m$ (typically $\sqrt{d}$ for classification, $d/3$ for regression), tree depth.

### AdaBoost

Sequential ensemble: each model $f_t$ focuses on the examples misclassified by the previous ensemble. Sample weights are updated multiplicatively after each round:

$$w_i^{(t+1)} \propto w_i^{(t)} \cdot \exp(-\alpha_t y_i f_t(\mathbf{x}_i))$$

where $\alpha_t = \frac{1}{2} \ln\frac{1 - \varepsilon_t}{\varepsilon_t}$ is the weight of model $t$ and $\varepsilon_t$ is its weighted error.

**Inductive bias:** Reduces bias progressively; sensitive to outliers (noisy labels get high weight).

### XGBoost (Extreme Gradient Boosting)

Gradient boosting with second-order Taylor expansion of the loss, regularised objective, and several systems-level optimisations (column subsampling, parallel split finding, cache-aware access):

$$\mathcal{L}^{(t)} = \sum_i \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

where $g_i = \partial_{\hat{y}} \ell(y_i, \hat{y})$ and $h_i = \partial^2_{\hat{y}} \ell(y_i, \hat{y})$ are first and second derivatives of the loss.

**Key hyperparameters:** Learning rate $\eta$, max depth, subsample ratio, $\lambda$ (L2 regularisation), $\alpha$ (L1 regularisation), number of rounds.

### Heterogeneous Ensembles and Classifier Combination

While homogeneous ensembles (Bagging, Boosting, Random Forest) combine multiple instances of the **same** base learner type, **heterogeneous ensembles** combine fundamentally different classifiers — e.g., a Decision Tree, a k-NN, and a Naive Bayes — exploiting the diversity of their distinct inductive biases.

**Motivation:** Classifiers that make different types of errors on different regions of the feature space are better candidates for combination than classifiers with correlated errors. Diversity is the key ingredient.

#### Combination Rules (Fixed)

Given $T$ classifiers $\{f_1, \ldots, f_T\}$ and $K$ classes, fixed combination rules aggregate outputs without learning additional parameters:

| Rule | Input type | Formula / logic |
|---|---|---|
| **Majority vote** | Class labels | $\hat{y} = \arg\max_k \sum_t \mathbb{1}[f_t(\mathbf{x}) = k]$ |
| **Weighted vote** | Class labels | $\hat{y} = \arg\max_k \sum_t w_t \cdot \mathbb{1}[f_t(\mathbf{x}) = k]$, $w_t$ set by validation accuracy |
| **Sum rule** | Posterior probabilities | $\hat{y} = \arg\max_k \sum_t P_t(k \mid \mathbf{x})$ |
| **Product rule** | Posterior probabilities | $\hat{y} = \arg\max_k \prod_t P_t(k \mid \mathbf{x})$ — assumes classifier independence |
| **Max rule** | Posterior probabilities | $\hat{y} = \arg\max_k \max_t P_t(k \mid \mathbf{x})$ |
| **Borda count** | Rankings | Sum of rank positions across classifiers |

**Sum rule** is empirically robust and theoretically well-motivated when classifiers produce calibrated probability estimates. **Product rule** is theoretically optimal under classifier independence (rarely holds in practice) but degrades badly when one classifier assigns near-zero probability.

#### Stacking (Stacked Generalisation)

A **trainable** combination method: a meta-learner $g$ is trained on the out-of-fold predictions of the base classifiers:

1. Split training data into $k$ folds.
2. For each fold, train all base classifiers on the remaining $k-1$ folds and collect predictions on the held-out fold.
3. Assemble a new dataset where each row is the vector of base classifier predictions; train $g$ on this dataset.
4. At test time: obtain base classifier predictions, feed to $g$ for the final prediction.

Stacking can learn non-linear combinations and asymmetric weighting — at the cost of added complexity and risk of overfitting the meta-learner.

#### Classifier Selection vs. Fusion

- **Fusion:** All classifiers contribute to every prediction (rules above).
- **Selection:** For a given query $\mathbf{x}$, select the single most competent classifier based on local accuracy in the neighbourhood of $\mathbf{x}$ (Dynamic Classifier Selection, DCS) or combine a subset (Dynamic Ensemble Selection, DES).

**Common pitfall:** Combining classifiers that are highly correlated (e.g., five Random Forests trained on the same features) yields negligible improvement — diversity is the prerequisite for ensemble gain.

---

### MLP as a Shallow Baseline

A Multilayer Perceptron with one or two hidden layers appears in this module as a **transition model** — shallow in the sense that feature engineering is still largely manual, but introducing the parametric learning paradigm that Module 5 develops deeply.

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \hat{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

**Key hyperparameters:** Layer sizes, activation function ($\sigma$: ReLU, tanh, logistic), learning rate, regularisation (L2, dropout), batch size, epochs.

---

## Classifier / Regressor Summary

| Method | Type | Inductive Bias | Key Hyperparameters |
|---|---|---|---|
| k-NN | Both | Smoothness, locality | $k$, distance metric |
| Naive Bayes | Classifier | Feature independence given class | Smoothing $\alpha$ |
| SVM / SVR | Both | Maximum margin / $\varepsilon$-tube | $C$, kernel, $\gamma$, $\varepsilon$ |
| Decision Tree | Both | Axis-aligned splits | Max depth, min samples |
| Random Forest | Both | Bagged decorrelated trees | $T$, $m$, depth |
| Bagging | Both | Variance reduction via bootstrap | $T$, base learner |
| AdaBoost | Classifier | Sequential bias reduction | $T$, base learner |
| XGBoost | Both | Regularised gradient boosting | $\eta$, depth, $\lambda$, $\alpha$ |
| MLP | Both | Universal approximation | Architecture, $\eta$, regularisation |

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapters 3–7 (linear models, kernel methods, sparse models).
- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley-Interscience, 2000. — Chapters 4–8.
- BREIMAN, L. Random forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.
- FREUND, Y.; SCHAPIRE, R. E. A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, v. 55, n. 1, p. 119–139, 1997.
- CHEN, T.; GUESTRIN, C. XGBoost: A scalable tree boosting system. In: *KDD*, 2016. p. 785–794. Available at: <https://arxiv.org/abs/1603.02754>.

---

## Extended Reading

- FRIEDMAN, J.; HASTIE, T.; TIBSHIRANI, R. *The Elements of Statistical Learning*. 2. ed. Springer, 2009. — Freely available at <https://hastie.su.domains/ElemStatLearn/>. Chapters 9–10 (trees and boosting), 12 (SVM).
- VAPNIK, V. N. *The Nature of Statistical Learning Theory*. 2. ed. Springer, 2000. — Original SVM theoretical foundation.
- SCHÖLKOPF, B.; SMOLA, A. J. *Learning with Kernels*. MIT Press, 2002. — Comprehensive kernel methods reference.

---

## Connected Activities

- [Atividade 1 — Part A](../activities/atividade-1.md#part-a--classification): All nine classifiers from this module (Decision Tree, k-NN, Naive Bayes, SVM, MLP, Random Forest, Bagging, AdaBoost, XGBoost).
- [Atividade 1 — Part B](../activities/atividade-1.md#part-b--regression): Seven regressors (Decision Tree, k-NN, SVR, MLP, Random Forest, Bagging, XGBoost).

---

*[← Module 2 — Experimental Protocols](02-protocols.md) · [README](../README.md) · [Module 4 — Descriptive Models →](04-descriptive.md)*
