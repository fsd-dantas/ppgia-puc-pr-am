# Module 5 — Deep Techniques

> **Colour:** ![#58a6ff](https://placehold.co/12x12/58a6ff/58a6ff.png) Blue  
> **Prerequisites:** [Module 1 — Foundations](01-foundations.md) · [Module 3 — Shallow Techniques](03-shallow.md) (MLP baseline)  
> **Connects to:** [Module 2 — Experimental Protocols](02-protocols.md) (evaluation applies unchanged)

---

## Overview

Deep techniques extend the parametric learning paradigm of the MLP to architectures capable of learning hierarchical feature representations directly from raw data. The central insight is that useful features — edges, textures, syntax, temporal patterns — need not be hand-crafted; they can be learned jointly with the classifier or regressor through backpropagation across many layers.

This module covers the biological motivation and the Perceptron as the atomic unit of neural computation, the theoretical foundations common to all deep architectures, and five canonical families: Multilayer Perceptrons (fully connected), Convolutional Neural Networks (spatial/local structure), Recurrent Neural Networks (sequential structure), Siamese Networks (similarity learning), and the Transformer with its role as a generative model.

---

## 2. Universal Approximation and the Case for Depth

A single hidden-layer MLP with a non-polynomial activation function can approximate any continuous function on a compact set to arbitrary precision — the **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991). This establishes theoretical expressiveness but does not address efficiency.

**Depth vs. width:** Representing certain functions requires exponentially wide shallow networks but only polynomially deep networks. Empirically, depth confers better generalisation on structured data (images, text, time-series) by composing progressively abstract representations.

---

## 3. Backpropagation and Gradient-Based Optimisation

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

## 4. Multilayer Perceptron (MLP)

A fully connected feedforward network. Each neuron in layer $l$ is connected to every neuron in layer $l-1$.

**Design decisions:** Number of layers, units per layer, activation function, output activation (softmax for multiclass, sigmoid for binary, linear for regression), loss function.

**Common pitfall:** More capacity is not always better — without regularisation, deeper/wider networks overfit on small datasets. The bias–variance trade-off from Module 1 applies directly.

---

## 5. Convolutional Neural Networks (CNN)

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

## 6. Recurrent Neural Networks (RNN)

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

## 7. Siamese Networks

A Siamese network is an architecture designed for **similarity learning**: rather than classifying an input into a fixed set of classes, it learns a function that measures how similar two inputs are. The architecture consists of two (or more) identical subnetworks — sharing the same weights — that each encode one input into an embedding space:

$$f_\theta(\mathbf{x}_1), \quad f_\theta(\mathbf{x}_2) \quad \text{(shared encoder } f_\theta \text{)}$$

The similarity (or distance) between the two embeddings drives the final decision. Common distance functions: Euclidean distance $\|f_\theta(\mathbf{x}_1) - f_\theta(\mathbf{x}_2)\|_2$, cosine similarity.

### Loss Functions

**Contrastive loss** (Hadsell et al., 2006) — pulls similar pairs together, pushes dissimilar pairs apart:

$$\mathcal{L} = (1 - y) \cdot \frac{1}{2} D^2 + y \cdot \frac{1}{2} \max(0,\, m - D)^2$$

where $D = \|f_\theta(\mathbf{x}_1) - f_\theta(\mathbf{x}_2)\|_2$, $y = 1$ for dissimilar pairs, and $m$ is a margin.

**Triplet loss** (Schroff et al., 2015) — anchor $\mathbf{x}_a$, positive $\mathbf{x}_p$ (same class), negative $\mathbf{x}_n$ (different class):

$$\mathcal{L} = \max\left(0,\; \|f_\theta(\mathbf{x}_a) - f_\theta(\mathbf{x}_p)\|_2^2 - \|f_\theta(\mathbf{x}_a) - f_\theta(\mathbf{x}_n)\|_2^2 + \alpha\right)$$

where $\alpha > 0$ enforces that positives are strictly closer than negatives by a margin.

### Applications

Siamese networks excel in **few-shot learning** and **metric learning** scenarios where class distributions shift or new classes appear at test time. The model generalises by similarity, not by memorised class boundaries:

- **Face verification** (same person / different person) — LFW benchmark.
- **Signature verification** and handwritten character recognition — central to the research group of Prof. Alceu (cf. Tannugi, Britto & Koerich, SMC 2019 on cross-dataset facial expression recognition).
- **Cross-dataset transfer** — the shared encoder can be pre-trained on one domain and fine-tuned for another.

**Key design decision:** The backbone encoder $f_\theta$ can be any architecture — CNN for images, LSTM for sequences, MLP for tabular data. The Siamese structure is a training paradigm, not a fixed architecture.

---

## 8. Transformer and Generative Models

The transformer (Vaswani et al., 2017) represents a fundamental shift from recurrent and convolutional inductive biases to a fully attention-based mechanism. It is the dominant architecture for sequence modelling, NLP, and increasingly for vision and multimodal tasks.

### Self-Attention Mechanism

Given an input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$, self-attention projects it into query, key, and value matrices via learned projections $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

The $\frac{1}{\sqrt{d_k}}$ scaling prevents dot products from saturating the softmax as $d_k$ grows. **Multi-head attention** runs $h$ independent attention heads in parallel, concatenates their outputs, and projects the result — allowing simultaneous attention to different representation subspaces.

**Key advantage over RNNs:** All positions are processed in parallel; long-range dependencies are modelled in $O(1)$ sequential operations (vs. $O(n)$ for RNNs). Trade-off: $O(n^2)$ memory and compute in sequence length $n$.

### Architecture

The original encoder–decoder transformer:

- **Encoder:** $N$ identical layers, each with multi-head self-attention + position-wise feed-forward + residual connection + layer normalisation.
- **Decoder:** Same structure plus a cross-attention sub-layer attending to encoder output.
- **Positional encoding:** Self-attention is permutation-invariant, so positional information is injected via sinusoidal encodings added to input embeddings.

### Transformer as a Generative Model

Generative transformers model the conditional distribution $p(x_t \mid x_1, \ldots, x_{t-1})$ autoregressively — the probability of the next token given all previous tokens. Generation proceeds by iterative sampling.

**Decoder-only (GPT family):** Trained on language modelling; generates sequences token by token.

**Encoder–decoder (T5, original seq2seq):** Encoder builds a context representation; decoder generates conditioned on it.

Broader generative model landscape:

| Model | Generative mechanism |
|---|---|
| Autoregressive transformer | $p(x_t \mid x_{<t})$ — sequential sampling |
| Variational Autoencoder (VAE) | Encoder maps to latent distribution $q(\mathbf{z}\mid\mathbf{x})$; decoder samples |
| GAN | Generator vs. discriminator adversarial training |
| Diffusion model | Iterative denoising of Gaussian noise toward data distribution |

**Graph Neural Networks (GNNs)** extend deep learning to graph-structured data — directly relevant to SDN topology modelling. They are surveyed in Extended Reading below.

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
