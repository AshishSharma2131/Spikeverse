# Spikeverse: A Neuromorphic Reinforcement Learning Framework for Autonomous Control

**Brain and Cognitive Science (BCS) Club, IIT Kanpur**[cite: 1]  
**Research Timeline:** May 2025 – Jul 2025[cite: 1]  
**Mentors:** Gaurav Rampuria, Kshitiz Tyagi, Saubhagya Pandey[cite: 1]

---

## Abstract

This repository details the research and implementation of **Spikeverse**, an advanced framework exploring the integration of Spiking Neural Networks (SNNs) with Deep Reinforcement Learning (DRL)[cite: 1]. The primary objective is to execute an Artificial Neural Network (ANN) to SNN conversion for game-playing agents, specifically targeting the Atari Breakout environment[cite: 1]. By translating dense, continuous neural representations into sparse, asynchronous spiking events, this architecture mitigates the extreme computational and energetic demands of traditional GPU-bound AI[cite: 1]. The converted neuromorphic agent achieves an 80% reduction in inference compute costs while preserving better-than-human performance and demonstrating high robustness to input perturbations[cite: 1].

---

## 1. Introduction & Motivation

The computational cost of training and deploying state-of-the-art Deep Neural Networks has grown exponentially, heavily reliant on dense matrix operations that trigger the von Neumann bottleneck[cite: 1]. Neuromorphic systems offer a biologically inspired, event-driven paradigm where computation is only activated when discrete spikes occur, drastically reducing energy consumption[cite: 1]. 

This project bridges these two domains: taking a high-performing Deep Q-Network (DQN) policy and migrating it to an SNN to study performance trade-offs, computational efficiency, and alternative neuron models[cite: 1].

---

## 2. Theoretical Framework: The RL Engine

Reinforcement Learning is mathematically formulated as a Markov Decision Process (MDP), defined by the 5-tuple $\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma)$[cite: 1]. The foundation of our learning engine leverages Q-learning, which relies on the Bellman Equation to iteratively estimate optimal state-action values[cite: 1]:

$$Q^{*}(s,a)=E_{s^{\prime}}[r+\gamma \max_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime})|s,a]$$

To achieve the necessary policy robustness in a sparse-reward environment like Atari Breakout, we engineered a **Rainbow DQN**, unifying six independent algorithmic advancements[cite: 1]:

| Optimization Module | Theoretical Contribution & Implementation |
| :--- | :--- |
| **Double DQN (DDQN)** | Decouples action selection (online network) from evaluation (target network) to mitigate the systematic overestimation bias inherent to standard Q-learning[cite: 1]. |
| **Prioritized Experience Replay** | Replaces uniform sampling with stochastic prioritization, assigning sampling probability $P(i)$ proportional to the absolute Temporal Difference (TD) error, combined with importance sampling weights to correct bias[cite: 1]. |
| **Dueling Architecture** | Decomposes the Q-value into independent streams for State Value $V(s)$ and Advantage $A(s,a)$, improving evaluation in states where individual action choices have minimal impact on the outcome[cite: 1]. |
| **N-Step TD Learning** | Computes the return $G_{t}^{(n)}$ by looking $n$ steps ahead, effectively propagating delayed rewards backward and optimizing credit assignment in sparse reward spaces[cite: 1]. |
| **Noisy Networks** | Injects factorized Gaussian noise into the dense layer parameters, allowing the agent to dynamically learn and attenuate exploration strategies rather than relying on heuristic epsilon-greedy schedules[cite: 1]. |
| **Distributional RL** | Models the complete probability distribution of returns $Z(s,a)$ rather than a single scalar expectation, capturing variance and multi-modal reward patterns[cite: 1]. |

---

## 3. Neuromorphic Architecture: SNN Conversion

Once the Rainbow DQN converges on an optimal continuous policy, the weights are mapped to a Spiking Neural Network[cite: 1]. This phase replaces standard artificial neurons (e.g., ReLU) with **Leaky Integrate-and-Fire (LIF)** models, which are governed by Ordinary Differential Equations (ODEs)[cite: 1].

### 3.1 The LIF Neuron Dynamics
The LIF neuron integrates incoming weighted spikes over time while simulating membrane leakage, analogous to an RC circuit[cite: 1]. The dynamics are described by:

$$\tau \frac{dv(t)}{dt} = -(v(t) - v_{rest}) + \sum_{i=1} W_i \text{Input}_i$$

When the membrane potential $v(t)$ crosses a defined firing threshold $v_{thresh}$, a discrete spike is emitted, and the potential resets[cite: 1]. Information in this network is propagated purely via spike timing and firing rates rather than activation magnitudes[cite: 1].

### 3.2 Weight Normalization
To ensure the successful transfer of the policy, the continuous ANN weights must be normalized[cite: 1]. Because spiking events become increasingly sparse in deeper layers of the network, we applied a scaling factor of 10 at each successive layer to maintain activation viability and optimize the sparse representation[cite: 1].

### 3.3 Spike-Timing Dependent Plasticity (STDP)
For localized learning, we also explore STDP, where synaptic weights $\Delta w$ are adjusted based on the timing difference $\Delta t = t_{post} - t_{pre}$[cite: 1]. This enables biologically plausible Long-Term Potentiation (LTP) and Long-Term Depression (LTD) based purely on causality[cite: 1].

---

## 4. Optimization under Non-Differentiability

The fundamental barrier to optimizing SNNs is the spike generation mechanism, formalized as a Heaviside step function $H(x)$[cite: 1]:

$$S(t)=H(u(t)-\theta)$$

Because the derivative of $H(x)$ is zero almost everywhere, standard gradient descent (backpropagation) fails[cite: 1]. 

### Surrogate Gradient Approximations
To resolve this, we implemented **Surrogate Gradients**[cite: 1]. During the forward pass, the network accurately outputs the non-differentiable binary spike[cite: 1]. During the backward pass, the gradient is approximated using smooth, differentiable surrogate functions that localize the derivative around the threshold[cite: 1]. We evaluated multiple surrogate derivatives, including[cite: 1]:

*   **Fast Sigmoid Derivative:** $\frac{1}{(1+|x|)^{2}}$[cite: 1]
*   **Exponential Function:** $e^{-|x|}$[cite: 1]
*   **Piece-wise Linear Function:** $\max(0,1-|x|)$[cite: 1]
*   **Cosh-based Function:** $\frac{1}{\cosh^{2}(x)}$[cite: 1]

These localized approximations ensure stable gradient flow, allowing the SNN to refine its policy representations.

---

## 5. Empirical Outcomes

The deployment of the Spikeverse architecture yielded critical validations for the future of neuromorphic autonomous systems:

1.  **Computational Efficiency:** The event-driven processing of the LIF network slashed inference compute costs by **80%** compared to the dense ANN baseline[cite: 1].
2.  **Performance Retention:** Despite the aggressive quantization into binary spikes and the sparsity of activations, the agent successfully retained better-than-human gameplay performance in Atari Breakout[cite: 1].
3.  **Adversarial Robustness:** The converted SNN displayed validated, high robustness to input perturbations, proving the inherent noise-tolerance of rate-coded spike networks in complex control environments[cite: 1].
