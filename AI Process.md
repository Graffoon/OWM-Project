This part is split to two section as two main models were used GEMENI and Claude.

# Working with AI — Project Process Log: The Genesis
## Orthogonal Weights Memory (OWM) Implementation

---

## Overview

This document describes the foundational stages of developing a Continual Learning neural network. It details the initial theoretical discussions, the decision to bypass common regularisation methods in favour of Orthogonal Weights Memory (OWM), and the step-by-step construction of the core architecture from scratch. It highlights the mathematical roadblocks encountered during the initial translation from whitepaper theory to working PyTorch code.

---

## Stage 1 — The Theoretical Foundation & Choosing OWM

The project began by establishing the physics of Catastrophic Forgetting. Standard neural networks are discriminative; when training on a new task, backpropagation blindly overwrites existing weights to minimize the immediate error gradient. 

We discussed the common solutions to this problem and explicitly rejected them:
* **Experience Replay:** Ruled out because storing old data violates strict data privacy and memory constraints.
* **Elastic Weight Consolidation (EWC):** Ruled out because it only applies "soft" penalties to weight changes. Over a long sequence of tasks, the penalties compound, and the network still degrades.

**The Unique Solution:**
We opted for Orthogonal Weights Memory (OWM). Instead of penalizing weight changes, OWM actively intercepts the error gradients and forces them into a mathematical null space. We established the core projection formula that would dictate the entire architecture:

$$\mathbf{g}_{projected} = P \cdot \mathbf{g}_{original}$$

* $\mathbf{g}_{projected}$: The modified gradient that will actually be applied to the weights.
* $P$: The projector matrix, acting as a dynamic notch filter.
* $\mathbf{g}_{original}$: The raw error gradient calculated by PyTorch.

---

## Stage 2 — Building from Scratch (The Component Architecture)

With the theory locked, we built the project completely from scratch rather than modifying an existing repository. We needed a system where the $P$ matrix existed on the GPU but was hidden from PyTorch's default optimizer.

**What we built:**
1.  **`OWMLayer`:** A custom neural network layer. Alongside standard weights and biases, we initialized $P$ as an Identity matrix ($I$). We used PyTorch's `register_buffer` to ensure $P$ was maintained in memory without being treated as a learnable parameter.
2.  **The Recursive Least Squares (RLS) Engine:** We implemented the specific mathematical update to "carve out" the memory shield after every batch:

$$P_{new} = P_{old} - rac{P_{old}x_{mean}x_{mean}^T P_{old}}{ lpha + x_{mean}^T P_{old}x_{mean}}$$

* $P_{old}$: The current state of the projector matrix.
* $x_{mean}$: The batch-averaged input vector representing the data signal.
* $ lpha$: The regularization scalar preventing division by zero and smoothing the update.

**The Overfit Test:**
Before touching real datasets, we wrote a sterile test loop (`main.py`). We fed the network a single, static batch of random noise for 100 iterations.
* *Success Criteria:* We tracked the trace (sum of the diagonal) of $P$. It successfully dropped from 784.0, proving the algorithm was mapping the space. The loss dropped to zero, proving the gradients were successfully passing through the identity matrix.

---

## Stage 3 — The Transition to Real Data (The Math Roadblocks)

Once the architecture was wired, we introduced the Split MNIST dataset (digits 0-4, then 5-9). This is where the translation from pure math to applied engineering hit its first major snags.

**Roadblock 1: The Low-Pass Filter Leak**
When Task 1 finished and Task 2 began, Task 1 accuracy dropped from ~95% to ~81%. 
* *The Cause:* We were taking the mean of 64 images ($x_{mean}$) to update $P$. This approximation smoothed out the high-frequency spatial details, leaving microscopic gaps in the memory shield that Task 2's gradients leaked through.
* *The Attempted Fix:* We changed the code to loop through all 64 images individually.
* *The Consequence:* Matrix Depletion. Subtracting from $P$ tens of thousands of times per epoch drove the matrix to zero. The network paralyzed itself because all gradients were multiplied by zero. We reverted to the batch mean and accepted the 81% retention as a functional success compared to the 0% baseline.

**Roadblock 2: The Optimizer Collision**
We attempted to tighten the filter by lowering $ lpha$ from 1.0 to 0.1. The loss curve violently exploded.
* *The Cause:* PyTorch's Stochastic Gradient Descent utilizes an internal Momentum buffer ($v_t$). PyTorch calculates the raw gradient, *we* project it through $P$, but then PyTorch adds the unprojected historical momentum directly into the weights, completely bypassing our middleware.
* *The Fix:* We strictly zeroed out PyTorch's Momentum and Weight Decay hyperparameters, forcing the system to rely purely on state-less gradient descent. 

---

## Summary of What the AI Got Right in the Early Stages

* **Architectural Segregation:** Correctly identifying that $P$ must be registered as a buffer, not a parameter, preventing PyTorch from destroying the matrix during backpropagation.
* **The Overfit Strategy:** Pausing to test the raw tensor flow with a dummy batch before introducing the massive MNIST dataset, isolating mathematical bugs from data-pipeline bugs.
* **Diagnostic Breakdowns:** Explaining *why* the matrix collapsed when we removed the batch mean, using system dynamics and filter analogies to make the abstract tensor math concrete.

## Summary of What Went Wrong in the Early Stages

* **Premature Optimization:** Attempting to tighten the $ lpha$ parameter before fully auditing PyTorch's internal optimizer mechanics, leading to the momentum-induced loss explosion.
* **Assuming Optimizer Compliance:** Failing to immediately recognize that PyTorch's default SGD implementation applies its regularizers *after* manual gradient adjustments, allowing inductive leaks.

## Overall Reflection

The most critical phase of the project was bridging the gap between the whitepaper mathematics and the framework's (PyTorch) hidden execution order. The theoretical formulas for OWM are flawless on paper, but implementing them required hacking the framework's native behavior. 

The process proved that when building custom middleware for neural networks, understanding what the framework does *implicitly* (like momentum buffering) is far more important than what it does explicitly. By stripping the system down to pure, un-optimized gradient descent, we were able to isolate the OWM variable and successfully defend the legacy weights.
