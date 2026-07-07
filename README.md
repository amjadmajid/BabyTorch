<div align="center"> <img alt="BabyTorch Logo" src="/images/babyTorchLogo.jpg">

<h3> Look under the hood, modify core algorithms, and when ready remove the word `baby` to work with PyTorch - <span style="font-style:italic; font-size:20px; font-weight:normal">BabyTorch's vision</span> </h3>
</div>

## Introduction

BabyTorch is a lightweight, educational deep learning framework that mirrors the PyTorch API with a minimal, readable implementation — small enough to read in an afternoon, capable enough to train a small GPT. It runs on CPU (NumPy) out of the box and on NVIDIA GPUs (CuPy) with zero code changes. Everything you learn here transfers directly to PyTorch.

```python
import babytorch
import babytorch.nn as nn
from babytorch.optim import SGD

x = babytorch.randn(32, 10, requires_grad=True)
model = nn.Sequential(nn.Linear(10, 32, nn.ReLU()), nn.Linear(32, 1))
loss = ((model(x) - 1) ** 2).mean()
loss.backward()                      # gradients for every parameter, automatically
```

## Installation

```bash
git clone https://github.com/amjadmajid/BabyTorch.git
cd BabyTorch
pip install -e .                     # CPU only -- NumPy is the sole dependency
```

Optional extras:

```bash
pip install -e ".[viz]"             # loss curves + computation-graph drawing
pip install -e ".[gpu]"             # GPU acceleration via CuPy (CUDA 12.x)
pip install -e ".[dev]"             # everything plus pytest
```

### Choosing a device (CPU / GPU)

BabyTorch runs on the CPU out of the box on any platform — Linux, macOS, Windows. With the `[gpu]` extra installed (NVIDIA GPU, CUDA 12.x), it picks the GPU automatically. Three ways to control the choice:

```python
import babytorch
babytorch.set_device("cpu")      # in code: "cpu", "cuda", or "auto"
```

```bash
BABYTORCH_DEVICE=cpu python train.py     # environment variable (initial device)
python train.py --device cpu            # CLI flag on the BabyGPT scripts
```

Pick the device **before** building tensors or models — arrays don't migrate between libraries after creation. There is no other GPU-specific code to learn: every module does its math through a single `xp` alias that resolves to NumPy or CuPy (see [`babytorch/backend.py`](babytorch/backend.py)).

**macOS note:** Macs have no CUDA, so BabyTorch runs on the CPU there (everything works, just slower for the bigger models). An Apple-Silicon GPU backend via MLX is on the roadmap — see `TODO.md`.

## The Book

The repository ships with a short book that explains the whole codebase in order — how a framework works, then how a GPT is built with it:

* **[Part I — The engine](book/README.md):** tensors, autograd, neural networks, training.
* **[Part II — BabyGPT](book/README.md):** tokenization, attention, the Transformer, pretraining/finetuning/generation.

Each chapter links to the exact source files it explains. Start at [`book/README.md`](book/README.md).

## Tutorials

Runnable, commented examples, from a two-line regression to a working language model:

1. **[BabyGPT — a tiny LLM](tutorials/llm/README.md)**: pretrain a decoder-only Transformer on Shakespeare, finetune it on nursery rhymes, and generate text. The flagship tutorial.
2. **[Regression](tutorials/regression/README.md)**: fit a noisy line/curve with a small MLP.
3. **Classification**: [binary](tutorials/classification/binary_classification/README.md), [multi-class](tutorials/classification/multi-class_classification/simple_multi-class_classification/README.md), and [MNIST digits with linear or convolutional models](tutorials/classification/multi-class_classification/mnist_digits_classification/README.md).

## Features

BabyTorch mirrors PyTorch's package structure:

- `babytorch.engine` — the autograd engine: `Tensor` plus every operation's forward *and* backward pass.
- `babytorch.nn` — layers (`Linear`, `Embedding`, `LayerNorm`, `Dropout`, `Conv2D`, ...), activations (`ReLU`, `GELU`, ...), losses (`MSELoss`, `CrossEntropyLoss`), and `nn.functional`.
- `babytorch.optim` — `SGD` (momentum, weight decay), `Adam`, `AdamW`, and LR schedulers including `CosineWarmupLR`.
- `babytorch.text` — `CharTokenizer` and a readable `BPETokenizer` (the GPT-family algorithm).
- `babytorch.datasets` — `DataLoader`, MNIST, and the Tiny Shakespeare corpus.
- `babytorch.visualization` — loss curves and rendering of the actual computation graph.
- `babytorch.backend` — the NumPy/CuPy device selection.

## Tests

The test suite is the ground truth that the framework works — and a good source of usage examples ([`tests/README.md`](tests/README.md)):

```bash
pip install -e ".[dev]"
pytest                        # full suite on CPU
BABYTORCH_DEVICE=cuda pytest  # the same suite on the GPU
```

Highlights: every differentiable op is checked against finite-difference gradients (`tests/test_autograd.py`), training tests prove an MLP solves XOR and a tiny GPT overfits a sequence (`tests/test_training.py`), and `tests/test_pytorch_parity.py` compares numbers against PyTorch when it is installed.

## Architecture design

The framework is built around one separation of concerns, kept everywhere:

1. **Engine** (`engine/`) — `operations.py` implements each operation's forward and backward math on raw arrays; `tensor.py` implements the `Tensor` data structure, records the computation graph, and replays it in reverse in `backward()`. *Math in operations, bookkeeping in tensors.*
2. **Neural networks** (`nn/`) — a ~100-line `Module` base class discovers parameters by walking attributes; layers are small compositions of tensor operations, so no layer needs custom gradient code.
3. **Optimizers** (`optim/`) — `step()`/`zero_grad()` over a parameter list; schedulers adjust the learning rate over time.
4. **Data** (`datasets/`, `text/`) — batching, standard datasets, and tokenizers that turn text into token ids.
5. **Visualization** (`visualization/`) — plot losses, or draw the recorded computation graph of any tensor.

### Directory structure

```bash
.
├── README.md
├── TODO.md
├── babytorch
│   ├── backend.py            # NumPy or CuPy ("xp"), chosen once
│   ├── engine
│   │   ├── operations.py     # forward + backward of every op
│   │   └── tensor.py         # Tensor, computation graph, backward()
│   ├── nn                    # Module, layers, losses, functional
│   ├── optim                 # SGD, Adam, AdamW, LR schedulers
│   ├── text                  # CharTokenizer, BPETokenizer
│   ├── datasets              # DataLoader, MNIST, Tiny Shakespeare
│   └── visualization         # loss curves, graph drawing
├── book                      # the BabyTorch book (Parts I and II)
├── tests                     # pytest suite (CPU and GPU)
└── tutorials
    ├── classification        # binary, multi-class, MNIST
    ├── regression            # linear/MLP regression
    └── llm                   # BabyGPT: pretrain -> finetune -> generate
```

## Contributing

We welcome contributions — BabyTorch favors readable implementations over fast ones, so the bar for a change is "does this make the idea clearer?". Check `TODO.md` for open tasks or propose your own.

## License

This project is licensed under the [MIT License](LICENSE).

---

Happy Learning! 🚀
