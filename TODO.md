## TODO
1. [x] Further develop the `nn.functional` module
2. [x] Change `model.zero_grad()` to `optimizer.zero_grad()` to be consistent with PyTorch (both exist; docs and tutorials use the optimizer form)
3. [x] Support GPU (CuPy backend, selected via `BABYTORCH_DEVICE` — see `babytorch/backend.py`)
4. [x] Add regression examples
5. [ ] Implement a few reinforcement learning algorithms
6. [ ] Add a KV cache to BabyGPT's `generate` so inference doesn't recompute past positions (a nice exercise after Chapter 8 of the book)
7. [ ] Add an attention-visualization example: plot a trained head's (T, T) weights as a heatmap
8. [ ] Apple-Silicon GPU backend via MLX, as a third library behind the `xp` proxy (macOS currently runs on the CPU; MLX's API differs from NumPy in places — scatter-add, random, dtypes — so this needs real porting and testing on a Mac)
