# Module 4 — Descriptive Models

> **Colour:** ![#bc8cff](https://placehold.co/12x12/bc8cff/bc8cff.png) Violet  
> **Prerequisites:** [Module 1 — Foundations](01-foundations.md) (linear algebra, probability, distance metrics)  
> **Connects to:** [Module 3 — Shallow Techniques](03-shallow.md) · [Module 5 — Deep Techniques](05-deep.md)

---

## Overview

Descriptive (unsupervised) models discover structure in data without labelled examples. Rather than optimising a prediction target, they seek to reveal the geometry of the data distribution: compact groups (clustering), low-dimensional manifolds (dimensionality reduction), and latent generative factors (density estimation).

These techniques are foundational in exploratory data analysis, feature engineering, data compression, and as pre-processing stages for supervised pipelines. In the research context of this study plan, unsupervised methods are directly applicable to network traffic segmentation, anomaly detection without labelled attack data, and topology analysis in SDN environments.

---

## 1. Clustering

Clustering partitions a dataset $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^n$ into groups such that intra-cluster similarity is high and inter-cluster similarity is low. The definition of "similarity" is determined by the algorithm's inductive bias.

### k-Means

Minimises total within-cluster variance (sum of squared distances to centroids):

$$\min_{\{\mu_k\}, \{C_k\}} \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

**Algorithm (Lloyd's):**
1. Initialise $K$ centroids (random, or k-means++ for better initialisation).
2. Assign each point to its nearest centroid.
3. Recompute centroids as the mean of assigned points.
4. Repeat steps 2–3 until convergence.

**Key hyperparameters:** $K$ (number of clusters), initialisation strategy, distance metric.

**Common pitfalls:** Sensitivity to initialisation (mitigated by k-means++, multiple restarts); assumes spherical clusters of equal size; $K$ must be specified a priori. The **elbow method** and **silhouette score** are standard heuristics for selecting $K$.

### Hierarchical Clustering

Builds a dendrogram — a tree of nested clusters — without requiring $K$ in advance. Two strategies:

- **Agglomerative (bottom-up):** Start with $n$ singleton clusters; merge the two most similar clusters at each step.
- **Divisive (top-down):** Start with one cluster containing all points; recursively split.

**Linkage criteria** define how inter-cluster distance is computed:

| Linkage | Distance between clusters $A$, $B$ |
|---|---|
| Single | $\min_{a \in A, b \in B} d(a, b)$ — susceptible to chaining |
| Complete | $\max_{a \in A, b \in B} d(a, b)$ — compact, sensitive to outliers |
| Average (UPGMA) | $\frac{1}{|A||B|} \sum_{a,b} d(a,b)$ — balanced |
| Ward | Minimises total within-cluster variance after merge |

### DBSCAN

Density-based clustering: clusters are contiguous regions of high density, separated by low-density regions.

- **Core point:** has at least $\text{MinPts}$ neighbours within radius $\varepsilon$.
- **Border point:** within $\varepsilon$ of a core point but not itself a core point.
- **Noise point:** neither core nor border — treated as outlier.

**Key hyperparameters:** $\varepsilon$ (neighbourhood radius), $\text{MinPts}$ (density threshold).

**Advantages over k-means:** Discovers arbitrary cluster shapes; automatically identifies outliers; does not require $K$.

### Cluster Validity Indices

| Index | Type | Formula sketch | Interpretation |
|---|---|---|---|
| Silhouette | Internal | $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$ | $\in [-1, 1]$; higher is better |
| Davies-Bouldin | Internal | Mean ratio of within-cluster to between-cluster distance | Lower is better |
| Adjusted Rand Index | External | Agreement with ground truth labels, corrected for chance | $\in [-1, 1]$; 1 is perfect |

---

## 2. Dimensionality Reduction

High-dimensional data is expensive to model, visualise, and store. Dimensionality reduction maps $\mathbf{x} \in \mathbb{R}^d$ to $\mathbf{z} \in \mathbb{R}^q$ with $q \ll d$, ideally preserving the structure relevant to downstream tasks.

### Principal Component Analysis (PCA)

Finds the $q$ orthogonal directions of maximum variance in the data. Given centred data matrix $\mathbf{X}$, the covariance matrix $\mathbf{C} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X}$ is decomposed as $\mathbf{C} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$. The first $q$ eigenvectors (columns of $\mathbf{V}$) define the projection:

$$\mathbf{Z} = \mathbf{X} \mathbf{V}_q$$

**Proportion of variance explained** by component $j$: $\frac{\lambda_j}{\sum_k \lambda_k}$. The **scree plot** shows this as a function of $j$.

**Key hyperparameters:** $q$ (number of components) or explained variance threshold (e.g., 95%).

**Common pitfall:** PCA is a linear method — it cannot capture non-linear manifold structure. For non-linear data, t-SNE or UMAP are preferred.

### Linear Discriminant Analysis (LDA)

Supervised dimensionality reduction: finds the projection that maximises between-class scatter relative to within-class scatter.

$$\mathbf{w}^* = \arg\max_\mathbf{w} \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}$$

where $\mathbf{S}_B$ is the between-class scatter matrix and $\mathbf{S}_W$ is the within-class scatter matrix. The maximum number of discriminant components is $\min(d, K-1)$ where $K$ is the number of classes.

### t-SNE

Stochastic Neighbour Embedding with Student-$t$ kernel: a non-linear technique for 2D/3D visualisation of high-dimensional data. Constructs a probability distribution over pairwise distances in the original space and optimises a lower-dimensional embedding to match it via KL divergence minimisation.

**Key hyperparameters:** Perplexity (controls effective neighbourhood size), learning rate, number of iterations.

**Limitation:** Not suitable as a pre-processing step for supervised models — the embedding is not deterministic and does not generalise to new points. Use for exploration and visualisation only.

### UMAP

Uniform Manifold Approximation and Projection: a manifold learning method that is faster than t-SNE, scales to larger datasets, and (unlike t-SNE) can be used for supervised dimensionality reduction and generalises to new data.

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapter 9 (mixture models and EM), Chapter 12 (PCA and probabilistic PCA).
- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley-Interscience, 2000. — Chapter 10 (unsupervised learning and clustering).

---

## Extended Reading

- JAIN, A. K.; MURTY, M. N.; FLYNN, P. J. Data clustering: a review. *ACM Computing Surveys*, v. 31, n. 3, p. 264–323, 1999. — Comprehensive survey of clustering algorithms.
- ESTER, M. et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In: *KDD*, 1996. p. 226–231. — Original DBSCAN paper.
- VAN DER MAATEN, L.; HINTON, G. Visualizing data using t-SNE. *Journal of Machine Learning Research*, v. 9, p. 2579–2605, 2008. Available at: <https://www.jmlr.org/papers/v9/vandermaaten08a.html>.
- McINNES, L.; HEALY, J.; MELVILLE, J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv*, 2018. Available at: <https://arxiv.org/abs/1802.03426>.
- FRIEDMAN, J.; HASTIE, T.; TIBSHIRANI, R. *The Elements of Statistical Learning*. 2. ed. Springer, 2009. — Chapter 14 (unsupervised learning). Available at: <https://hastie.su.domains/ElemStatLearn/>.

---

## Connected Activities

No activity in the current course sequence directly targets Module 4. However, dimensionality reduction (PCA) is a natural pre-processing step for the classification task in Atividade 1 Part A and may be explored as an extension.

---

*[← Module 3 — Shallow Techniques](03-shallow.md) · [README](../README.md) · [Module 5 — Deep Techniques →](05-deep.md)*
