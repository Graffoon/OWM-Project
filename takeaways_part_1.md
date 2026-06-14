# Takeaways  Catastrophic Forgetting with OWM

## Why I Chose This Approach

I was tasked with finding a paper on catastrophic forgetting and a solution to it. Rather than going with the first result, I consulted several AI models to help me scout the space and deliberately looked for the most niche and unique approach I could find. OWM stood out. After landing on it, I spent a significant amount of time working through its implementation and building up my understanding of neural networks in general  it was not a light read, but that was part of the appeal.

## What I Understood by Implementing It

Two things became clear through the process of actually building this. First, that it is genuinely possible to overcome catastrophic forgetting without storing any additional data  not through brute force memory, but through clever mathematics that constrains the geometry of weight updates. That was satisfying to see work in practice.

The second thing I learned is arguably why OWM stays niche: it is simply not practical. The method is computationally inefficient, architecturally sensitive, and prone to failure modes that are difficult to debug. In an era where large data centers exist and replaybased methods are straightforward to implement, the overhead of maintaining and updating projector matrices per layer is hard to justify. Interesting in theory. but in my view, implementation of this method is best left in the theory realm.

## What Surprised Me

The failure mode of the 10task experiment was genuinely surprising. When I tried to more faithfully replicate the paper by splitting MNIST into 10 singledigit tasks, the results were bizarre  Task 1 at 40%, the last task at 100%, and everything in between at 0%. I expected either a graceful degradation or a clear improvement. Instead the setup was simply broken at an architectural level. The output layer has 10 neurons, and training each task on a single digit means nine of those neurons receive no gradient signal per task. By the end, only the last task's neuron is trained at all.

This taught me something important: experimental choices in papers carry implicit assumptions about architecture that are not always stated. The 2task split I originally designed was actually the correct choice for this network  not a simplification, but the right fit.

## On Using AI in This Project

AI usage during this project was substantial and genuinely fruitful  and since it was actively encouraged, there was no hesitation in leaning on it. The collaboration was most valuable when it came to explaining *why* something works rather than just generating code. The geometric explanation of how P's null space relates to network width, the confirmation that the batchmean projector update is mathematically valid, and catching the train/test split error early on were all genuinely useful contributions.

The one notable failure was when the AI implemented every paper detail at once  10 tasks, 10 runs, decaying alpha, dropout, momentum  without flagging that the architecture was incompatible with the singleclass task setup. Technically accurate advice produced a broken result. That experience is a good reminder that AI suggestions still need to be filtered through your own understanding of the problem.

## Closing Thought

Catastrophic forgetting is a surprisingly deep problem for something that looks simple on the surface. The fact that a neural network cannot sequentially learn two tasks without destroying the first feels almost embarrassing  humans do it effortlessly. OWM's answer, that the geometry of weight space can be managed so that new learning occupies directions old learning does not depend on, is one of the more satisfying ideas I encountered in this course. It does not fully solve the problem, but it frames it in a way that feels honest about what is actually happening inside the network.
