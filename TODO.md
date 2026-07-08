## TODO
1. [x] Further develop the `nn.functional` module
2. [x] Change `model.zero_grad()` to `optimizer.zero_grad()` to be consistent with PyTorch (both exist; docs and tutorials use the optimizer form)
3. [x] Support GPU (CuPy backend, selected via `BABYTORCH_DEVICE` — see `babytorch/backend.py`)
4. [x] Add regression examples
5. [x] Implement a few reinforcement learning algorithms (tabular DP/TD plus REINFORCE, Actor-Critic, DQN, PPO on GridWorld + Snake — see `tutorials/rl/` and book Part III, chapters 9–11)
6. [x] Add a KV cache to BabyGPT's `generate` so inference doesn't recompute past positions (on by default; `generate.py --no_cache` shows the difference)
7. [x] Add an attention-visualization example: `tutorials/llm/attention_viz.py` plots trained heads' (T, T) weights as heatmaps
8. [ ] Apple-Silicon GPU backend via MLX, as a third library behind the `xp` proxy (macOS currently runs on the CPU; MLX's API differs from NumPy in places — scatter-add, random, dtypes — so this needs real porting and testing on a Mac)
