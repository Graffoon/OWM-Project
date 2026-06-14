# Lessons Learned — Failed Experiments & Future Improvements

## Overview

Not everything tried in this project worked. This document covers the experiments that failed, why they failed, and — more usefully — what a next iteration of this project could do differently. Some of these ideas go beyond what was implemented, but they're worth recording as a roadmap for anyone (including future me) picking this back up.

---

## Part 1 — Failed Experiments

### 1.1 The 10-Task Single-Digit Split

**What we tried:** In an attempt to more faithfully reproduce the original OWM paper, the Split-MNIST setup was changed from 2 tasks (digits 0–4, then 5–9) to 10 tasks — one task per individual digit, trained sequentially. Alongside this, the paper's other specifications were added: SGD with momentum, L2 regularisation, dropout, and a decaying alpha schedule for the projector update.

**Results:**
<img width="1753" height="443" alt="image" src="https://github.com/user-attachments/assets/5f3c09df-0b4a-4fbe-8840-07d9b96d41e1" />

**Why it failed:** The network's output layer has 10 neurons — one logit per digit class. When a task contains only a single digit, the loss only ever provides a gradient signal to the one output neuron corresponding to that digit. The other nine neurons receive zero gradient for that entire task. After training sequentially through all 10 single-digit tasks, only the most recently trained neuron (digit 9) ends up with meaningful weights — everything else degrades toward whatever those neurons happened to settle at, which for most of them is effectively useless. Task 1 retains a partial 40% likely because the very first task at least had a chance to shape the network before being overwritten nine more times, and digit 0 may be geometrically distinct enough in pixel space to leave some residual signal.

This isn't a subtle OWM failure — it's a fundamental mismatch between the task structure and the architecture. OWM's gradient projection was working exactly as designed; it just had nothing useful to protect, because each task only ever trained a sliver of the network to begin with.

**A note on what got reverted alongside this:** The momentum, L2 regularisation, dropout, and decaying alpha additions were all rolled back together with the 10-task split, but none of them were actually the cause of the failure — they were collateral. These are genuinely worth revisiting independently in the 2-task setup (more on this in Part 2).

---

### 1.2 Aggressive Alpha Reduction

**What we tried:** To improve Task 1 retention after Task 2 training, alpha was lowered from 0.75 toward 0.1, and Task 1 training was extended to more epochs than Task 2. The reasoning was sound — a smaller alpha makes each RLS update larger, so P closes off Task 1's subspace more aggressively before Task 2 ever begins.

**Results:** Task 1 retention did improve, but Task 2 accuracy dropped noticeably — by enough that it was clearly a problem, not a minor tradeoff.

**Why it "failed":** This wasn't a bug — it's OWM's central tradeoff made visible. P has a fixed amount of "space" (determined by the hidden layer width) that can be allocated between protecting old tasks and leaving room for new ones. Pushing alpha down doesn't create more space; it just reallocates more of the existing space toward Task 1. With a hidden dimension of only 800, there wasn't enough room to give Task 1 strong protection without meaningfully starving Task 2.

This experiment was valuable precisely because it failed in the expected direction — it's a clean empirical demonstration of the theoretical tradeoff, and it's what motivated the move to a wider network instead.

---

## Part 2 — What I Would Do Differently

### 2.1 A Better Stress Test: Permuted MNIST

This connects directly back to 1.1. The 10-task experiment was trying to test something legitimate — how OWM behaves with more tasks and more diverse inputs — but it broke the architecture by changing the *output* structure per task.

**Permuted MNIST** is the standard continual learning benchmark that avoids this entirely. Each task still uses all 10 digit classes (0–9), so the output layer is fully utilised in every task — but the *pixels of every image are shuffled according to a fixed random permutation that's different per task*. The digits are now visually meaningless, but the classification problem is still 10-way.

This would have been the right way to test the criticism raised earlier — that Split-MNIST's two halves are too statistically similar to stress-test OWM's geometric limits. Permuted MNIST forces each task to occupy a genuinely different region of input space (since the permutation scrambles spatial structure entirely) while keeping the architecture's output requirements intact. You could run 5, 10, or even 20 permutation-tasks sequentially and watch exactly when P starts running out of room — without ever hitting the single-neuron starvation problem from 1.1.

### 2.2 Alternative Subspace Update Algorithms

The current implementation uses RLS (Recursive Least Squares) with a fixed alpha and a batch-mean approximation. A few directions worth exploring:

**Revisit the decaying alpha schedule independently.** The paper's formula `α_i = α₀ · λ^(i/n)` was reverted only because it was bundled with the 10-task change — it was never actually shown to be wrong. Re-implementing it in the working 2-task setup could improve results without any of the architectural issues.

**SVD-based subspace extraction (Gradient Projection Memory / GPM).** Instead of incrementally updating a projector matrix P via RLS, this family of methods collects a sample of activations after training each task, computes the SVD, and keeps the top-k singular vectors as an explicit basis for "directions this task used." Future gradients are projected to be orthogonal to the union of all stored bases. This is conceptually similar to OWM but trades the continuous online update for an exact, interpretable basis computed once per task — and it would let you directly visualise how many dimensions each task consumes, which ties nicely into the projector demo already built.

**Hybrid approaches.** Combining OWM's projection with a small experience replay buffer, or with an EWC-style penalty term, could offset OWM's stiffness — the projection prevents interference, while replay or the penalty term gives the network a gentle pull back toward retaining old-task performance even in directions P didn't fully block.

### 2.3 Additional Datasets Beyond MNIST

MNIST is forgiving — small, clean, low-resolution, and grayscale. A natural extension path, roughly in order of difficulty:

**Fashion-MNIST**, same format as MNIST (28×28 grayscale, 10 classes) but clothing items instead of digits — a drop-in replacement that requires zero architecture changes but tests whether OWM's behaviour generalises to a different (if structurally similar) dataset.

**Cross-domain sequences** — train Task 1 on MNIST digits and Task 2 on Fashion-MNIST items. This is the genuinely dissimilar-task setup that Split-MNIST doesn't provide. It would be the most direct test of whether OWM's protection holds up when the input distributions truly don't overlap.

**CIFAR-10/100** — color images, more classes, and critically, this would require moving from a flat MLP to a convolutional architecture. That's a substantial step up: P would need to operate on flattened convolutional feature maps or be reformulated per-channel, since the current `(input_size × input_size)` formulation assumes a fully-connected layer. Worth flagging as "future work" rather than something to casually bolt on.

### 2.4 Systematic Hyperparameter and Architecture Study

Rather than the ad-hoc tuning done here (try a value, see what happens), a more rigorous version would run a small grid:

- hidden_dim ∈ {800, 1200, 2000, 4000}
- alpha ∈ {0.01, 0.1, 0.5, 0.75, 1.0}

...and plot Task 1 retention vs Task 2 accuracy as a genuine tradeoff curve (a Pareto frontier) rather than single before/after numbers. This would turn "we tried lowering alpha and it hurt Task 2" into an actual quantitative relationship, and would let you point to a specific (hidden_dim, alpha) combination as the empirically-justified choice rather than an intuition-based one.

### 2.5 Better Experimental Methodology

A few smaller but meaningful upgrades:

**Continuous accuracy tracking.** Currently Task 1 accuracy is measured only before and after Task 2 training — two points. Measuring it every few batches *during* Task 2 training would produce an actual forgetting curve, showing whether the drop is gradual or sudden, and at what point during Task 2 training it happens.

**Multiple runs with mean ± std.** A single run with a fixed seed (as currently implemented) can't distinguish a genuine effect from noise. Running each configuration 5–10 times with different seeds and reporting mean ± std — without the 10-task architecture problem this time — would make any comparison between OWM and the baseline statistically meaningful.

**Per-task confusion matrices.** The current evaluation reports a single accuracy number. A confusion matrix for Task 1 after Task 2 training would show *which* digits get confused with *which* — potentially revealing whether forgetting is uniform across digits or concentrated on specific ones that overlap visually with Task 2's digits.

---

## Closing Thoughts

The two failures documented here — the 10-task collapse and the alpha/Task-2 tradeoff — both turned out to be informative rather than just dead ends. The 10-task failure exposed a real architectural constraint that's easy to miss when reading a paper in isolation, and directly points toward Permuted MNIST as the correct fix. The alpha tradeoff is OWM's central limitation made concrete, and motivates both the wider-network fix that was implemented and the more systematic hyperparameter study that wasn't.

If this project continues, Permuted MNIST plus a proper (hidden_dim, alpha) sweep with multiple runs would be the highest-value next steps — they directly address the two biggest open questions this project raised without requiring an architectural overhaul.
