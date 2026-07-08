## TODO
1. [x] Further develop the `nn.functional` module
2. [x] Change `model.zero_grad()` to `optimizer.zero_grad()` to be consistent with PyTorch (both exist; docs and tutorials use the optimizer form)
3. [x] Support GPU (CuPy backend, selected via `BABYTORCH_DEVICE` — see `babytorch/backend.py`)
4. [x] Add regression examples
5. [x] Implement a few reinforcement learning algorithms (tabular DP/TD plus REINFORCE, Actor-Critic, DQN, PPO on GridWorld + Snake — see `tutorials/rl/` and book Part III, chapters 9–11)
6. [x] Add a KV cache to BabyGPT's `generate` so inference doesn't recompute past positions (on by default; `generate.py --no_cache` shows the difference)
7. [x] Add an attention-visualization example: `tutorials/llm/attention_viz.py` plots trained heads' (T, T) weights as heatmaps
8. [x] Add diffusion models (Part IV): a 2-D toy DDPM and an MNIST U-Net on shared machinery (`tutorials/diffusion/`), with the one new op `nn.Upsample` and book chapters 12–13
9. [x] Render the book to PDF and add a full Arabic edition: `book/build.sh en` (pandoc + XeLaTeX) and `book/build.sh ar` (Markdown → HTML → WeasyPrint, RTL); a complete Arabic translation under `book/ar/` (all 13 chapters + Contents, code kept verbatim and checked by the drift-guard) reusing the English figures; see `book/BUILD.md`
10. [ ] Apple-Silicon GPU backend via MLX — experimental `mps` backend now scaffolded (`set_device("mps")`, adapter in `babytorch/mlx_backend.py`, `pip install -e ".[mlx]"`) but **untested on device**, since it was authored on an Intel Mac. Remaining: validate and fix on an M-series Mac. MLX's API differs from NumPy where it matters — float32-only (no float64, so the finite-difference gradient checks stay on CPU), key-based random, functional scatter-add, and array-method quirks — all flagged with `VERIFY` notes in the adapter
