# Preface

Deep learning is often taught from the outside: import a framework, assemble
layers, call `backward()`, and trust the machinery. That is useful for building
models, but it leaves a gap between the equations in a textbook and the system
that executes them. BabyTorch closes that gap with a deliberately small,
working framework whose important paths can be read end to end.

This is not a catalogue of every feature in PyTorch, nor a recipe for training
frontier models. It is a guided reconstruction of the durable ideas underneath
modern systems: tensors, reverse-mode automatic differentiation, trainable
modules, optimization, attention, reinforcement learning, and diffusion. Each
idea appears first as a mental model, then as mathematics and shapes, and
finally as executable code. The tests are part of the explanation: they show
what correctness means for gradients, layers, and complete training loops.

## Who this book is for

You should be comfortable reading Python and basic algebra. You do not need a
machine-learning course beforehand. The only calculus idea required at the
start is that a derivative measures how an output changes when an input is
nudged; Chapter 2 builds the rest when it is needed.

The book supports three reading paths:

1. **Framework path:** Chapters 1-4, then Appendix A. Choose this if you want
   to understand autograd, modules, optimizers, and the move to PyTorch.
2. **Language-model path:** Chapters 1-8. Choose this to build a small GPT from
   text encoding through cached generation.
3. **Generative-and-agent path:** Read Chapters 1-4 first, then continue with
   reinforcement learning (9-11) or diffusion (12-13).

## How to work through it

Read beside the repository. Install the package from the repository root with
`pip install -e .`, run every **Try it** block, and change one value before
moving on. Each chapter ends with short retrieval questions and a tested
implementation exercise. The fast questions check whether the model in your
head is sound; the implementation track turns that model into code.

Shapes are written in monospace, such as `(B, T, C)`, where `B` is batch size,
`T` is sequence length, and `C` is the feature width. Scalars and equations use
ordinary mathematical notation. Code uses BabyTorch names; Appendix A maps the
small API differences that matter when you move to PyTorch.

## Scope and honesty

BabyTorch prioritizes legibility over production performance. It uses one
process, one active array backend, and unfused operations. Production
frameworks add device-local tensors, compiled kernels, distributed execution,
mixed precision, graph optimization, and a much larger API. Those differences
matter operationally. They do not change the chain rule, matrix calculus,
attention equations, Bellman targets, or diffusion objective developed here.

The source code is the final authority. When the prose and implementation
could drift, automated tests compare the printed snippets with the real files.
If you find a mismatch or a clearer explanation, the project welcomes a small,
tested contribution.

