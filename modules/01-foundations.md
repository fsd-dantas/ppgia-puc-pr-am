# Module 1 — Foundations

> **Colour:** ![#3fb950](https://placehold.co/12x12/3fb950/3fb950.png) Green  
> **Prerequisites:** Calculus (derivatives, gradients), basic programming (Python/NumPy)  
> **Connects to:** [Module 2 — Experimental Protocols](02-protocols.md) · [Module 3 — Shallow Techniques](03-shallow.md)

---

## Overview

Foundations establishes the mathematical substrate on which all Machine Learning methods are built. Three pillars are covered: **linear algebra** (the language of data representation and transformations), **analytical geometry** (the spatial intuition behind distance, margin, and projection), and **probability & statistics** (the formal framework for uncertainty, inference, and model evaluation).

These are not prerequisites to be skimmed — they are the precise tools invoked when reading algorithm derivations, interpreting loss functions, or understanding why a method fails on a given dataset. Every subsequent module assumes fluency with the notation and results introduced here.

---

## 1. Linear Algebra

### Vectors and Matrices

A dataset of $n$ samples with $d$ features is represented as a matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, where each row $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector. A target vector $\mathbf{y} \in \mathbb{R}^n$ collects the labels or response values.

Key operations:

- **Matrix–vector product:** $\mathbf{Xw}$ computes a linear combination of features weighted by $\mathbf{w}$ — the core of every linear model.
- **Transpose:** $\mathbf{X}^\top \mathbf{X} \in \mathbb{R}^{d \times d}$ is the Gram matrix; its invertibility determines whether the normal equations have a unique solution.
- **Rank:** $\text{rank}(\mathbf{X}) < d$ signals multicollinearity — features are linearly dependent.

### Eigendecomposition

For a symmetric positive semi-definite matrix $\mathbf{A}$:

$$\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top$$

where $\mathbf{Q}$ contains orthonormal eigenvectors and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_d)$ contains non-negative eigenvalues. This decomposition underlies **Principal Component Analysis (PCA)** and the analysis of covariance structures.

### Singular Value Decomposition (SVD)

Any matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ decomposes as:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

where $\mathbf{U} \in \mathbb{R}^{n \times n}$, $\mathbf{\Sigma} \in \mathbb{R}^{n \times d}$ (diagonal), $\mathbf{V} \in \mathbb{R}^{d \times d}$. SVD is the numerical backbone of PCA, low-rank approximation, and pseudo-inverse computation.

### Norms

- **$\ell_2$ (Euclidean):** $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ — used in k-NN distance and SVM margin.
- **$\ell_1$ (Manhattan):** $\|\mathbf{x}\|_1 = \sum_i |x_i|$ — induces sparsity in Lasso regularisation.
- **Frobenius:** $\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ — matrix analogue of the $\ell_2$ norm.

**Common pitfall:** Failing to normalise features before computing distances causes high-magnitude features to dominate — directly affecting k-NN, SVM, and gradient-based methods.

---

## 2. Analytical Geometry

### Distance and Similarity

The choice of distance metric is a modelling decision, not a default. Common metrics:

| Metric | Formula | Typical use |
|--------|---------|-------------|
| Euclidean | $\sqrt{\sum_i (x_i - y_i)^2}$ | k-NN, k-means, SVM (RBF) |
| Manhattan | $\sum_i \|x_i - y_i\|$ | Robust to outliers; high-dimensional data |
| Cosine | $1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ | Text/sparse vectors; direction over magnitude |
| Mahalanobis | $\sqrt{(\mathbf{x}-\mathbf{y})^\top \mathbf{S}^{-1} (\mathbf{x}-\mathbf{y})}$ | Accounts for feature covariance |

### Hyperplanes and Margins

A hyperplane in $\mathbb{R}^d$ is defined by:

$$\mathbf{w}^\top \mathbf{x} + b = 0$$

Points on either side satisfy $\mathbf{w}^\top \mathbf{x} + b > 0$ or $< 0$ — this is the geometric basis of linear classifiers. The **margin** is the perpendicular distance between the hyperplane and the nearest data points, equal to $\frac{2}{\|\mathbf{w}\|_2}$ in the SVM formulation.

### Projections

The orthogonal projection of $\mathbf{x}$ onto a unit vector $\mathbf{u}$ is $(\mathbf{x}^\top \mathbf{u})\mathbf{u}$. PCA finds the directions $\mathbf{u}_1, \ldots, \mathbf{u}_k$ that maximise projected variance — reducing dimensionality while retaining maximal information.

---

## 3. Probability & Statistics

### Probability Fundamentals

- **Random variable:** A mapping from outcomes to real values. Discrete (Bernoulli, Categorical) or continuous (Gaussian, Exponential).
- **Joint, marginal, conditional:** $P(X, Y) = P(Y \mid X) \cdot P(X)$ — the chain rule underpins generative models.
- **Bayes' theorem:** $P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}$ — the foundation of Naive Bayes and Bayesian inference.

### Key Distributions

| Distribution | Parameters | ML relevance |
|---|---|---|
| Bernoulli | $p$ | Binary classification labels |
| Categorical | $\mathbf{p} \in \Delta^{K-1}$ | Multiclass labels; softmax output |
| Gaussian | $\mu, \sigma^2$ | Feature modelling; LDA; noise assumptions |
| Multivariate Gaussian | $\boldsymbol{\mu}, \boldsymbol{\Sigma}$ | Gaussian mixture models; QDA |

### Expectation and Variance

$$\mathbb{E}[X] = \int x \, p(x) \, dx \qquad \text{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$$

The **bias–variance decomposition** of the expected generalisation error of an estimator $\hat{f}$:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}^2[\hat{f}] + \text{Var}[\hat{f}] + \sigma^2_\varepsilon$$

where $\sigma^2_\varepsilon$ is irreducible noise. This identity is the theoretical motivation for regularisation, ensemble methods, and model complexity control.

### Maximum Likelihood Estimation (MLE)

Given i.i.d. observations $\{\mathbf{x}_i\}_{i=1}^n$ and a parametric model $p(\mathbf{x} \mid \boldsymbol{\theta})$:

$$\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \sum_{i=1}^n \log p(\mathbf{x}_i \mid \boldsymbol{\theta})$$

MLE is the derivation basis for cross-entropy loss (classification) and mean squared error (regression under Gaussian noise).

### Hypothesis Testing and Significance

When comparing classifier performance across multiple datasets or folds, statistical tests are required:

- **Paired t-test:** Appropriate for comparing two classifiers on the same dataset with k-fold CV results.
- **Wilcoxon signed-rank test:** Non-parametric alternative; preferred when normality of differences cannot be assumed.
- **Friedman test + Nemenyi post-hoc:** Standard for comparing multiple classifiers across multiple datasets.

**Common pitfall:** Reporting accuracy differences without a significance test overstates the evidence for superiority — a difference of 1–2% on a single dataset is rarely statistically meaningful.

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapters 1–2 (probability, decision theory) and Appendix C (linear algebra).
- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley-Interscience, 2000. — Chapter 2 (Bayesian decision theory).
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. — Part I: Applied Math and Machine Learning Basics (Chapters 2–5). Available at: <https://www.deeplearningbook.org>.

---

## Extended Reading

- STRANG, G. *Introduction to Linear Algebra*. 5. ed. Wellesley-Cambridge Press, 2016. — The definitive accessible treatment; Chapters 6–7 for eigenvalues and SVD.
- BLITZSTEIN, J. K.; HWANG, J. *Introduction to Probability*. 2. ed. CRC Press, 2019. — Rigorous but readable; freely available at <https://projects.iq.harvard.edu/stat110>.
- PETERSEN, K. B.; PEDERSEN, M. S. *The Matrix Cookbook*. Technical University of Denmark, 2012. — Dense reference for matrix identities and derivatives. Available at: <https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf>.
- GRINSTEAD, C. M.; SNELL, J. L. *Introduction to Probability*. American Mathematical Society, 2006. — Open access at <https://math.dartmouth.edu/~prob/prob/prob.pdf>.

---

## Connected Activities

No activity in this course directly targets Module 1 in isolation — foundational concepts are exercised implicitly through all experimental work. The following connections are most direct:

- [Atividade 1 — Part A](../activities/atividade-1.md#part-a--classification): Feature space geometry (distance metrics, decision boundaries) is operative in k-NN and SVM.
- [Atividade 1 — Part B](../activities/atividade-1.md#part-b--regression): Bias–variance trade-off is observable empirically across the regressor comparison.

---

*← [README](../README.md) · [Module 2 — Experimental Protocols →](02-protocols.md)*
