# Catastrophic Forgetting — Solved with OWM

## The Problem

When a neural network learns a new task sequentially, it tends to overwrite the weights it built for previous tasks. By the time Task 2 is done, Task 1 performance collapses — sometimes to near zero. This is **catastrophic forgetting**, and it is one of the core unsolved challenges in continual learning.

## The Solution

This project implements **Orthogonal Weights Modification (OWM)** (Zeng et al., 2019) — a continual learning method that constrains the *direction* of gradient updates rather than preventing them altogether. After each training batch, OWM records which directions in weight space the current task depends on, and ensures all future updates are orthogonal to those directions. New tasks can still learn freely — they just cannot touch what previous tasks already claimed.

The experiment uses **Split-MNIST**: the network is trained on digits 0–4 (Task 1), then on digits 5–9 (Task 2), and Task 1 accuracy is measured again afterward. A plain baseline network is trained identically but without OWM, showing the full extent of forgetting that OWM prevents.

## Results

<img width="1334" height="730" alt="owm_results" src="https://github.com/user-attachments/assets/d904742f-5cd4-4fb1-863c-747ae2823143" />

## File Structure

| File | Description |
|------|-------------|
| [`main.py`](main.py) | Entry point — runs training, evaluation, and produces the results chart |
| [`Network.py`](Network.py) | OWM layer and network architecture, including the projector matrix P and gradient projection logic |
| [`data_handler.py`](data_handler.py) | Downloads and filters MNIST into task-specific DataLoaders |
| [`algorithmic_thinking.md`](algorithmic_thinking.md) | Step-by-step breakdown of the algorithm, the math behind OWM, and all design decisions |
| [`demo.py`](demo.py) | A small demo that shows how the protection matrix works (requires numpy and matplotlib and is ran individually) |
| [`gemeni_log.md`](gemeni_log.md) | Full log of the initially AI-assisted development process — what worked, what failed, and how issues were resolved (Gemeni) |
| [`claude_log.md`](claude_log.md) | Full log of the finally AI-assisted development process — what worked, what failed, and how issues were resolved (Claude) |
| [`takeaways.md`](takeaways.md) | Personal reflections on the project |

## How to Run

```bash
pip install torch torchvision matplotlib
python main.py
```

MNIST will be downloaded automatically on the first run. Results are saved to `owm_results.png`.

## Reference

Zeng, G., Chen, Y., Cui, B., & Yu, S. (2019). *Continual Learning of Context-dependent Processing in Neural Networks*. Nature Machine Intelligence. https://arxiv.org/abs/1810.01256
