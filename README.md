# Algorithmic Thinking — Solving Catastrophic Forgetting with OWM

## The Problem: What is Catastrophic Forgetting?

When a neural network learns Task 2, it updates its weights to minimise the loss on Task 2's data. The problem is that the same weights were already tuned for Task 1 — and nothing stops the new updates from overwriting them. By the time Task 2 training is done, the network has effectively forgotten Task 1 ever existed.

This is **catastrophic forgetting**, and it is one of the core unsolved challenges in continual learning. A human can learn to ride a bike and later learn to drive a car without forgetting how to cycle. A standard neural network cannot do this — learning the car completely overwrites the bike.

The experiment here demonstrates this concretely using Split-MNIST: a network is trained on digits 0–4 (Task 1), then on digits 5–9 (Task 2), and Task 1 accuracy is measured again. Without protection, it collapses.

---

## The Approach: Orthogonal Weights Modification (OWM)

OWM is a continual learning method introduced by Zeng et al. (2019). The core idea is elegant: rather than preventing weight changes altogether, **constrain the direction of weight updates** so they do not interfere with what the network already knows.

Specifically, OWM ensures that when learning Task 2, every gradient update is **orthogonal to the input subspace of Task 1**. If a gradient points in a direction that Task 1 cares about, it gets projected away. Only the components that Task 1 is indifferent to are allowed through.

The method requires no storing of old data, no extra task-specific heads, and no retraining — making it one of the cleaner solutions in the space.

---

## Step-by-Step Breakdown

### Step 1 — The Network Architecture

The network is a two-layer MLP:

```
Input (784) → OWMLayer (800 hidden units) → ReLU → OWMLayer (10 outputs)
```

Each `OWMLayer` is a standard linear layer with one addition: a **projector matrix P** of shape `(input_size × input_size)`. This matrix is not a learnable parameter — the optimiser never touches it. It is updated manually using a separate rule.

P starts as the identity matrix `I`, meaning "no subspace has been claimed yet — all gradient directions are allowed."

### Step 2 — The Forward Pass

Nothing unusual here. Input passes through layer 1, ReLU is applied, then through layer 2 to produce logits. The only extra step is that the inputs to each layer are **saved** during the forward pass:

```python
self.input1 = x          # raw input to layer 1
self.h1 = relu(layer1(x)) # post-ReLU output, which is the input to layer 2
```

These cached values are needed later for the P update.

### Step 3 — Backpropagation

Loss is computed with cross-entropy and `.backward()` is called as normal. At this point, each weight tensor has a `.grad` populated by standard backprop. This is where OWM intervenes.

### Step 4 — Gradient Projection (The Core OWM Step)

Before the optimiser applies the gradients, they are intercepted and projected:

```
grad ← P @ grad^T  (then transposed back to match weight shape)
```

This operation rotates the gradient so that any component pointing into the previously-seen input subspace is removed. The gradient that actually reaches the optimiser only contains directions orthogonal to Task 1's inputs — directions Task 1 doesn't use and therefore won't be harmed by.

During Task 1, P = I, so the projection is a mathematical no-op (`I @ grad = grad`). The method only activates meaningfully from Task 2 onward.

### Step 5 — Weight Update

The optimiser (SGD) applies the projected gradient to update the weights. Because the gradient has been constrained, the update cannot damage the directions that encode Task 1's knowledge.

### Step 6 — Projector Update (The Memory Step)

After the weights are updated, P is tightened using the **Recursive Least Squares (RLS)** formula from the paper:

```
P ← P − (P x̄)(P x̄)ᵀ / (α + x̄ᵀ P x̄)
```

Where `x̄` is the mean input to the layer over the current batch, and `α` is a scalar that controls how aggressively P updates.

Intuitively: this is a rank-1 downdate that shrinks P in the direction of `x̄`. Every batch that passes through the layer leaves an impression on P — "this direction matters to me, block it for future tasks." Over a full task's training, P accumulates a memory of the entire input subspace that task used.

---

## The Projector Math — Intuition

Think of P as a projection onto the **null space** of all previously seen inputs. When P is applied to a gradient vector, it removes any component that points toward the subspace the network has already learned.

The RLS update is derived from the same mathematics used in adaptive filtering — it is the recursive version of computing a least-squares orthogonal projector from streaming data. Using the batch mean as the representative vector is a practical approximation: instead of updating P once per sample, one update per batch achieves similar coverage at lower cost.

The `α` term in the denominator serves two roles: it prevents division by zero, and it acts as a regulariser that slows down how quickly P shrinks. A smaller α → faster, more aggressive tightening → stronger protection of past tasks but less room for new ones. A larger α → more conservative updates → weaker protection but more plasticity.

---

## Experiment Design

**Dataset:** Split-MNIST — the standard MNIST digit dataset split into two sequential tasks.
- Task 1: digits 0, 1, 2, 3, 4
- Task 2: digits 5, 6, 7, 8, 9

**Evaluation protocol:**
1. Train on Task 1 → evaluate Task 1 accuracy (baseline)
2. Train on Task 2 → evaluate Task 2 accuracy, then re-evaluate Task 1 accuracy

The key metric is the **drop in Task 1 accuracy** after Task 2 training. A large drop = catastrophic forgetting. OWM should show a much smaller drop than the baseline.

**Baseline:** An identical MLP (same width, same depth) trained with plain SGD and no gradient projection. This is the control that demonstrates the problem OWM solves.

---

## Results Interpretation

The bar chart produced by the experiment shows three measurements for both models:

- **Task 1 accuracy right after Task 1 training** — both models should be comparable here, since OWM's projection is a no-op on the first task
- **Task 2 accuracy right after Task 2 training** — OWM may be slightly lower here because the projected gradients have less freedom
- **Task 1 accuracy after Task 2 training** — the key comparison: this is where OWM should clearly outperform the baseline

---

## Design Decisions and Tradeoffs

**Why OWM over other methods?**
Methods like EWC (Elastic Weight Consolidation) require computing and storing a Fisher information matrix. Replay-based methods require storing old training data. OWM needs neither — only the P matrix per layer, which grows quadratically in input size but is fixed once the layer dimensions are set. For the network sizes used here, this is entirely manageable.

**Why hidden size 800?**
This matches the architecture in the original paper's Split-MNIST experiments. Hidden size directly determines the size of P — a wider network gives P more null space to work with, leaving more room for future tasks after Task 1 has claimed its portion.

**Why batch-mean for the projector update?**
The paper describes two modes: sample-by-sample (expensive, more precise) and batch-mean (cheaper, approximately equivalent in practice). The batch-mean approach achieves comparable results at a fraction of the computational cost.

---

## References

Zeng, G., Chen, Y., Cui, B., & Yu, S. (2019). *Continual Learning of Context-dependent Processing in Neural Networks*. Nature Machine Intelligence, 1(8), 364–372. https://arxiv.org/abs/1810.01256
