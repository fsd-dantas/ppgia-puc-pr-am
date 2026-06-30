# Module 5 — Deep Techniques

> **Colour:** ![#58a6ff](https://placehold.co/12x12/58a6ff/58a6ff.png) Blue  
> **Prerequisites:** [Module 1 — Foundations](01-foundations.md) · [Module 3 — Shallow Techniques](03-shallow.md) (MLP baseline)  
> **Connects to:** [Module 2 — Experimental Protocols](02-protocols.md) (evaluation applies unchanged)

---

## Overview

Deep techniques extend the parametric learning paradigm of the MLP to architectures capable of learning hierarchical feature representations directly from raw data. The central insight is that useful features — edges, textures, syntax, temporal patterns — need not be hand-crafted; they can be learned jointly with the classifier or regressor through backpropagation across many layers.

This module covers the theoretical foundations common to all deep architectures, followed by the three canonical families: Multilayer Perceptrons (fully connected), Convolutional Neural Networks (spatial/local structure), and Recurrent Neural Networks (sequential structure). Modern variants and the current transformer-dominated landscape are surveyed as extended reading.

---

## 1. Universal Approximation and the Case for Depth

A single hidden-layer MLP with a non-polynomial activation function can approximate any continuous function on a compact set to arbitrary precision — the **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991). This establishes theoretical expressiveness but does not address efficiency.

**Depth vs. width:** Representing certain functions requires exponentially wide shallow networks but only polynomially deep networks. Empirically, depth confers better generalisation on structured data (images, text, time-series) by composing progressively abstract representations.

---

## 2. Backpropagation and Gradient-Based Optimisation

### Forward Pass

For a network with $L$ layers, the forward pass computes activations sequentially:

$$\mathbf{a}^{(l)} = \sigma\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l = 1, \ldots, L$$

### Backpropagation

Computes the gradient of the loss $\mathcal{L}$ with respect to all parameters via the chain rule. The gradient at layer $l$:

$$\boldsymbol{\delta}^{(l)} = \left(\mathbf{W}^{(l+1)\top} \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'\left(\mathbf{z}^{(l)}\right)$$

where $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ and $\odot$ denotes element-wise multiplication.

### Optimisers

| Optimiser | Update rule (sketch) | Notes |
|---|---|---|
| SGD | $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla \mathcal{L}$ | Baseline; needs tuned $\eta$ and momentum |
| Momentum | Accumulates gradient history | Dampens oscillations |
| RMSprop | Adapts learning rate per parameter | Effective for non-stationary objectives |
| Adam | Combines momentum + RMSprop | Default for most tasks; $\beta_1=0.9$, $\beta_2=0.999$ |

### Activation Functions

| Function | Formula | Range | Notes |
|---|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $(0,1)$ | Saturates; vanishing gradient in deep nets |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1,1)$ | Zero-centred; still saturates |
| ReLU | $\max(0, z)$ | $[0, \infty)$ | Default hidden activation; dead neuron problem |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \ll 1$ | $(-\infty, \infty)$ | Mitigates dead neurons |
| GELU | $z \cdot \Phi(z)$ | $(-\infty, \infty)$ | Used in transformers |

### Regularisation

- **L2 (weight decay):** Adds $\frac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss — penalises large weights.
- **Dropout:** Randomly zeroes activations with probability $p$ during training; approximates an ensemble of $2^n$ subnetworks.
- **Batch Normalisation:** Normalises layer inputs to zero mean and unit variance per mini-batch; accelerates training and acts as implicit regularisation.
- **Early stopping:** Monitors validation loss; halts training when it stops decreasing.

---

## 3. Multilayer Perceptron (MLP)

A fully connected feedforward network. Each neuron in layer $l$ is connected to every neuron in layer $l-1$.

**Design decisions:** Number of layers, units per layer, activation function, output activation (softmax for multiclass, sigmoid for binary, linear for regression), loss function.

**Common pitfall:** More capacity is not always better — without regularisation, deeper/wider networks overfit on small datasets. The bias–variance trade-off from Module 1 applies directly.

---

## 4. Convolutional Neural Networks (CNN)

CNNs exploit the spatial locality and translational invariance of image-like data through **convolutional layers**:

$$(\mathbf{X} * \mathbf{K})_{i,j} = \sum_m \sum_n \mathbf{X}_{i+m, j+n} \cdot \mathbf{K}_{m,n}$$

**Key components:**

- **Convolutional layer:** Learns local feature detectors (kernels/filters). Parameters: kernel size, number of filters, stride, padding.
- **Activation (ReLU):** Applied element-wise after convolution.
- **Pooling layer:** Downsamples spatial dimensions — max pooling retains the most salient activation in each region.
- **Fully connected layer:** Aggregates learned features for the final prediction.

**Parameter sharing** (same kernel applied across all spatial positions) drastically reduces parameter count compared to a fully connected layer on raw pixels.

**Landmark architectures:** LeNet-5, AlexNet, VGG, ResNet (residual connections to mitigate vanishing gradients), EfficientNet.

---

## 5. Recurrent Neural Networks (RNN)

RNNs process sequential data by maintaining a hidden state $\mathbf{h}_t$ that summarises history:

$$\mathbf{h}_t = \sigma(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})$$

**Vanishing / exploding gradient problem:** Gradients propagated over long sequences either vanish (exponentially small) or explode (exponentially large), making plain RNNs impractical for long-range dependencies.

### Long Short-Term Memory (LSTM)

Introduces a cell state $\mathbf{c}_t$ and three gating mechanisms:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

The cell state acts as a long-term memory that can be selectively read, written, and erased — resolving the vanishing gradient problem for sequences of practical length.

### Gated Recurrent Unit (GRU)

A simplified variant of LSTM with two gates (reset and update) and no separate cell state. Competitive performance with fewer parameters.

### Research Application (SDN / Smart Grid)

LSTMs and GRUs are directly applicable to the time-series prediction tasks in the author's MSc research:
- **VNF auto-scaling:** Predict future traffic load from historical SDN flow statistics → LSTM regressor.
- **Smart Grid forecasting:** Predict power demand from historical consumption and weather data → seq2seq LSTM.
- **Anomaly detection:** Model normal traffic sequences; flag deviations — unsupervised LSTM autoencoder.

---

## 6. Beyond: Transformers and Modern Architectures

The transformer architecture (Vaswani et al., 2017) has largely supplanted RNNs for sequence modelling. Self-attention computes pairwise relevance across all positions simultaneously:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**Graph Neural Networks (GNNs)** extend deep learning to graph-structured data — directly relevant to SDN topology modelling.

These architectures are surveyed as extended reading; they are not covered in the course's primary scope.

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. — Part II: Modern Practical Deep Networks (Chapters 6–12). Available at: <https://www.deeplearningbook.org>.
- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapter 5 (neural networks).
- LECUN, Y.; BENGIO, Y.; HINTON, G. Deep learning. *Nature*, v. 521, p. 436–444, 2015.

---

## Extended Reading

- HOCHREITER, S.; SCHMIDHUBER, J. Long short-term memory. *Neural Computation*, v. 9, n. 8, p. 1735–1780, 1997. — Original LSTM paper.
- VASWANI, A. et al. Attention is all you need. In: *NeurIPS*, 2017. Available at: <https://arxiv.org/abs/1706.03762>. — Transformer architecture.
- HE, K. et al. Deep residual learning for image recognition. In: *CVPR*, 2016. Available at: <https://arxiv.org/abs/1512.03385>. — ResNet; residual connections.
- SCARSELLI, F. et al. The graph neural network model. *IEEE Transactions on Neural Networks*, v. 20, n. 1, p. 61–80, 2009. — GNN foundations; relevant to SDN topology.
- ZHANG, C. et al. A deep neural network for traffic flow prediction. *IEEE Transactions on Intelligent Transportation Systems*, 2019. — Time-series deep learning applied to network traffic; methodological parallel to VNF load prediction.

---

## Connected Activities

No course activity currently targets Module 5 exclusively. The MLP appears in [Atividade 1](../activities/atividade-1.md) as a shallow-regime baseline. Deep architectures will be exercised in future activities.

---

*[← Module 4 — Descriptive Models](04-descriptive.md) · [README](../README.md)*
