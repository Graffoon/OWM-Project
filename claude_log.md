# Working with AI — Project Process Log
## Catastrophic Forgetting with OWM

---

## Overview

This document describes the process of developing the OWM continual learning project with the help of Claude (Anthropic). It covers what worked, what failed, how we recovered, and what the collaboration actually looked like in practice — including the moments where following the AI's suggestions made things worse.

---

## Stage 1 — Initial Code Review

The project started with a working OWM implementation across three files: `data_handler.py`, `Network.py`, and `main.py`. The core math — the RLS projector update and gradient projection — was already in place and mostly correct.

The first prompt was an open-ended review request: *"tell me what you think, what could use a touch up, what's already fine."*

**What the AI caught:**
- The evaluation was running on the training split instead of the held-out test split — a fundamental methodological error that would make results look better than they are
- The device was hardcoded to `'cuda'`, which crashes on any machine without a GPU
- There was no baseline model to compare against, meaning the OWM results had no context
- There was no graph output, which the assignment explicitly requires

**What the AI correctly left alone:**
- The RLS projector update math was confirmed correct
- The gradient projection formula was confirmed correct
- The 2-task Split-MNIST setup was kept as-is

**Result:** Clean, minimal improvements — test split added, device detection fixed, a `BaselineNetwork` class added, and a bar chart added to `main.py`. The code ran correctly and produced meaningful results.

---

## Stage 2 — Consulting the Paper (Where Things Went Wrong)

After sharing the original OWM paper (Zeng et al. 2019), the AI was asked whether there was anything noteworthy to replicate. This is where the first significant failure happened.

The AI identified several legitimate differences between the code and the paper:
- The paper uses **momentum SGD**, not plain SGD
- The paper uses **L2 regularisation** (coefficient 0.001)
- The paper uses **dropout** (rate 0.2)
- The paper's main result is a **10-task experiment** (one digit per task), not a 2-task split
- The paper reports **mean ± std across 10 repeated runs**
- The paper specifies a **decaying alpha** for batch-wise projector updates

All of these points were technically accurate. The AI implemented all of them at once.

**What broke and why:**

The 10-task setup was the critical mistake. The network has 10 output neurons — one per digit class — which works fine when both tasks together cover all 10 classes (0–4 and 5–9). But when each task contains only a single digit, the network only ever receives loss signal for one output neuron per task. The other nine neurons go untrained. By the time all 10 tasks have been learned sequentially, the results looked like this:

- Task 1 (digit 0): ~40% accuracy
- Task 10 (digit 9): ~100% accuracy
- All other tasks: ~0% accuracy

This is not catastrophic forgetting in the interesting sense — it is a broken experimental setup. The architecture was never designed for single-class tasks; each task needs multiple classes so the output layer is fully utilised.

The added complexity (10 runs, decaying alpha, dropout, momentum, L2) also made the code substantially longer and harder to follow, which was the opposite of what was needed.

**The correction:**

The AI was told directly: *"these changes only messed up the code"* and *"it's best we stick with what I already wrote on my own but with a bit of modification."*

This was the right call. We rolled back to the 2-task split (digits 0–4 vs digits 5–9), stripped out all the extra complexity, and returned to the minimal working version. The only changes kept from Stage 2 were:
- The `use_owm` flag in `train_task()` so one function handles both models
- The test/train split separation
- The baseline comparison and bar chart

**Lesson:** The paper's experimental setup is not always directly portable to a student implementation. Matching every detail of a paper without understanding the architectural assumptions behind each choice can break a working system. The 2-task setup was the correct choice for this network — the AI should have flagged the incompatibility rather than implementing it blindly.

---

## Stage 3 — Tuning OWM Performance

With the working 2-task setup restored, the next question was how to improve Task 1 retention after Task 2 training without hurting Task 2 too much.

**Suggestions made:**
1. Lower alpha (e.g. 0.1 or 0.01) — makes P tighten more aggressively, blocking Task 1's subspace more thoroughly before Task 2 starts
2. More training epochs on Task 1 only — more P updates fire before Task 2 sees any data
3. Experience replay — mixing a few Task 1 batches into Task 2 training

The first two were tried. The outcome: Task 1 retention improved, but Task 2 accuracy dropped noticeably. This is the fundamental OWM tradeoff — the more aggressively you protect past tasks, the less freedom the network has to learn new ones.

**The actual fix — bigger hidden layer:**

The AI then explained why increasing the hidden layer size addresses this tradeoff structurally rather than just shifting it. The projector matrix P has shape `(input_size × input_size)`. For a hidden layer of size 800, P is an 800×800 matrix. As Task 1 trains, P progressively blocks off directions in this space. By the time Task 2 starts, there may not be much orthogonal space left for new updates to move through freely.

Increasing the hidden dimension (e.g. to 2000 or 4000) gives P a much larger null space. Task 1 claims its portion, but significantly more room remains for Task 2 to learn without being over-constrained by the projection. The protection on Task 1 stays just as strong — there is simply more geometry to work with.

```python
owm_model = OWMNetwork(784, 2000, 10).to(DEVICE)
```

This was flagged as the cleanest solution because it does not require changing any algorithm parameters or training logic — just one number.

---

## Stage 4 — Documentation

The final stage was producing the written deliverables.

The `algorithmic_thinking.md` document was generated in one pass and needed no corrections. Because the AI had been working on the code throughout the project, it already had a full picture of every design decision — why OWM was chosen over EWC or replay methods, how the RLS update works mathematically, why the batch-mean approximation is used, what the tradeoff of alpha represents.

For inline code comments, the first attempt was misunderstood — the AI added comments to the code snippets inside the markdown document instead of to the actual `.py` files. Once clarified, the correct files were produced with a comment on every non-trivial line.

---

## Summary of What the AI Got Right

- Identifying the train/test split error immediately
- Confirming the OWM math was correct and leaving it alone
- Keeping the code minimal and close to the original structure after the rollback
- Explaining the geometric reason why a wider network helps OWM specifically
- Producing the algorithmic thinking document accurately in one pass

## Summary of What Went Wrong

- Implementing the full paper setup (10 tasks, 10 runs, decaying alpha) without flagging that the architecture was incompatible with single-class tasks
- Significantly increasing code complexity in a direction the project did not need
- Initially adding comments to the wrong place (markdown snippets instead of source files)

## Overall Reflection

The most valuable moments in this collaboration were not when the AI generated things from scratch, but when it explained *why* something works or fails. The geometric explanation of why a wider network helps OWM, or the mathematical confirmation that the batch-mean projector update is a valid approximation — those were more useful than any code it wrote.

The most dangerous moment was Stage 2, where technically accurate information led to a broken implementation. AI suggestions should be evaluated against the actual constraints of the project, not accepted because they are grounded in a legitimate source.
