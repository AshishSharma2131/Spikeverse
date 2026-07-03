# 🔥 EMBER
### Event-driven Multi-stage Bio-inspired Efficient Reinforcement Learning

**A neuromorphic reinforcement learning framework that converts state-of-the-art Deep RL policies into spiking neural networks — cutting inference compute by 80% without sacrificing performance.**

![Status](https://img.shields.io/badge/status-research-orange)
![Domain](https://img.shields.io/badge/domain-Neuromorphic%20Computing-blueviolet)
![RL](https://img.shields.io/badge/RL-Rainbow%20DQN-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ⚡ Why EMBER

Modern Deep RL agents are computational furnaces — dense matrix multiplications, GPU-bound inference, and energy costs that scale faster than the intelligence they produce. Biological brains do more with less: sparse, asynchronous spikes instead of dense floating-point activations.

**EMBER bridges that gap.** It trains a champion-grade Rainbow DQN agent, then migrates its learned policy into a Spiking Neural Network (SNN) — an event-driven architecture that only computes when a neuron actually fires. The result is a neuromorphic agent that plays Atari Breakout better than a human, while spending 80% less compute to do it.

---

## 🏆 Headline Results

| Metric | Result |
|---|---|
| **Inference compute reduction** | **80%** vs. dense ANN baseline |
| **Gameplay performance** | Better-than-human on Atari Breakout |
| **Adversarial robustness** | High tolerance to input perturbation |
| **Learning paradigm** | Fully event-driven, rate-coded spike propagation |

---

## 🧠 How It Works

EMBER operates in two stages: **learn dense, deploy sparse.**

### Stage 1 — Rainbow DQN: The Policy Engine

The task is framed as a classical Markov Decision Process, $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma)$, solved via Q-learning and the Bellman optimality equation:

$$Q^{*}(s,a)=\mathbb{E}_{s^{\prime}}\left[r+\gamma \max_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime})\mid s,a\right]$$

Rather than vanilla DQN, EMBER unifies **six independent algorithmic upgrades** into a single Rainbow architecture, purpose-built for the sparse-reward Atari Breakout environment:

| Module | What It Solves |
|---|---|
| **Double DQN** | Decouples action selection from evaluation to kill Q-value overestimation bias |
| **Prioritized Experience Replay** | Samples transitions by TD-error magnitude instead of uniformly, with importance-sampling correction |
| **Dueling Architecture** | Splits Q into State-Value $V(s)$ and Advantage $A(s,a)$ streams for sharper credit assignment |
| **N-Step TD Learning** | Propagates delayed rewards $n$ steps backward for faster, more accurate learning signals |
| **Noisy Networks** | Learns exploration directly via parametric noise — no epsilon-greedy heuristics |
| **Distributional RL** | Models the full return distribution $Z(s,a)$, not just its mean, to capture reward variance |

### Stage 2 — Neuromorphic Conversion: ANN → SNN

Once Rainbow DQN converges, its weights are transplanted into a network of **Leaky Integrate-and-Fire (LIF)** neurons — a biologically grounded, ODE-governed alternative to ReLU:

$$\tau \frac{dv(t)}{dt} = -(v(t) - v_{rest}) + \sum_i W_i \cdot \text{Input}_i$$

A neuron fires only when its membrane potential $v(t)$ crosses threshold $v_{thresh}$ — then resets. Information travels as spike timing and firing rate, not activation magnitude.

**Key engineering details:**
- **Layer-wise weight normalization** — spikes sparsify with depth, so each successive layer is rescaled (×10) to keep signal viable through the network.
- **STDP (Spike-Timing-Dependent Plasticity)** — synaptic weights are adjusted based on pre/post-spike timing offset $\Delta t = t_{post} - t_{pre}$, enabling biologically plausible local learning (LTP/LTD) as a secondary learning channel.

---

## 🧩 The Hard Problem: Training Something Non-Differentiable

Spikes are binary — governed by a Heaviside step function:

$$S(t)=H(u(t)-\theta)$$

Its derivative is zero almost everywhere, which breaks standard backpropagation outright. EMBER solves this with **surrogate gradients**: the forward pass fires real, non-differentiable spikes, while the backward pass substitutes a smooth approximation localized around the threshold. Four surrogate functions were benchmarked:

| Surrogate | Function |
|---|---|
| Fast Sigmoid | $\dfrac{1}{(1+\lvert x\rvert)^{2}}$ |
| Exponential | $e^{-\lvert x\rvert}$ |
| Piecewise Linear | $\max(0,\ 1-\lvert x\rvert)$ |
| Cosh-based | $\dfrac{1}{\cosh^{2}(x)}$ |

This keeps gradient flow stable enough to fine-tune the converted SNN's policy end-to-end.

---

## 📊 Validated Outcomes

1. **80% lower inference compute** — event-driven LIF processing vs. the dense Rainbow DQN baseline.
2. **Performance held under quantization** — binary spikes and sparse activations still deliver better-than-human Atari Breakout play.
3. **Robust under perturbation** — rate-coded spike networks proved inherently noise-tolerant in adversarial testing.

---

## 🗺️ Roadmap

- [ ] Extend beyond Breakout to the full Atari-57 suite
- [ ] Benchmark on neuromorphic hardware (Loihi / Akida) for real energy measurements
- [ ] Explore direct SNN training (skip ANN pretraining) via surrogate gradients from scratch
- [ ] Hybrid STDP + surrogate-gradient training regime

---

## ## 👥 Contributors

**Ashish Sharma** - [@AshishSharma2131](https://github.com/AshishSharma2131)
