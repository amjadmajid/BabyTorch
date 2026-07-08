# Reinforcement Learning with BabyTorch

Everything so far has learned from a **fixed dataset**. Reinforcement
learning is different: there is no dataset. An **agent** acts in a game,
the game returns **rewards**, and the agent has to discover -- by trial
and error -- which actions pay off. No labels, no answer key; just a
score to maximise.

This tutorial ports the deep-RL agents from
[**deep-reinforcement-learning-games-from-scratch**](https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch)
onto BabyTorch. That is a deliberate demonstration: those agents are
written in PyTorch, and moving them here is almost a search-and-replace
(`torch.nn` → `babytorch.nn`, `torch.optim.Adam` → `babytorch.optim.Adam`).
The one thing BabyTorch lacks -- `torch.distributions.Categorical` -- is a
[six-line helper](common.py). If you know how to build a classifier in
BabyTorch, you already know how to build a policy.

## The four algorithms

Two great families of RL, and the modern method that tops each:

| Script | Idea | Family |
|--------|------|--------|
| [`reinforce.py`](reinforce.py) | Make actions that led to high returns more likely. | policy gradient |
| [`actor_critic.py`](actor_critic.py) | Same, but a learned `V(s)` supplies a per-state baseline (the *advantage*). | policy gradient |
| [`dqn.py`](dqn.py) | Learn `Q(s,a)` -- the value of each action -- and act greedily on it. | value-based |
| [`ppo.py`](ppo.py) | Actor-critic with a *clipped* update, reused for several epochs. The modern workhorse (and the "PO" in RLHF). | policy gradient |

The supporting cast:

| File | What it holds |
|------|---------------|
| [`gridworld.py`](gridworld.py) | The GridWorld maze -- a Gym-style `reset`/`step` environment, pure NumPy. |
| [`snake.py`](snake.py) | The game of Snake, with feature-vector *or* raw-grid observations. |
| [`snake_dqn.py`](snake_dqn.py) | `dqn.py` pointed at Snake -- with an MLP or a ConvNet. |
| [`common.py`](common.py) | The shared machinery: `Categorical`, the networks, returns/GAE, the replay buffer, Huber loss. |

## Quickstart

```bash
pip install -e .          # from the repo root, once

cd tutorials/rl
python reinforce.py        # policy gradient on GridWorld
python actor_critic.py     # + a learned value baseline
python dqn.py              # value-based, with replay + a target network
python ppo.py              # the clipped, sample-efficient one
```

Each script trains, prints its progress, saves a learning-curve PNG, and
(for the value-based agents) reports how its **greedy** policy does. Force
the device with `--device cpu` / `--device cuda`, or
`BABYTORCH_DEVICE=cpu python ppo.py`.

### GridWorld: all four solve it

The default maze has an optimal path worth a return of about **+2** (a
short walk to the goal, +10 bonus minus the steps); a lost episode scores
about **-100**. All four agents get there, but not equally fast:

| Agent | Solves GridWorld | Note |
|-------|------------------|------|
| REINFORCE | ~70 updates | pure policy gradient, highest variance |
| Actor-Critic | ~60 updates | the value baseline steadies it |
| DQN | greedy **100%** | off-policy; learns from replayed experience |
| **PPO** | **~15 updates** | clipping + GAE + epoch reuse = far fewer samples |

Watch an agent think in ASCII with `python reinforce.py --render_every 20`,
or make the maze harder by editing `MAZES["classic"]` in `gridworld.py`.

## Snake: the same agent, a bigger game

Snake is `dqn.py` again -- only the game changes. The agent sees the board
in one of two ways:

```bash
python snake_dqn.py --net features     # 11-number summary -> tiny MLP (fast)
python snake_dqn.py --net grid --rows 6 --cols 6 --shaping 0.5   # raw board -> ConvNet
```

* **`--net features`** feeds the DQN an 11-number summary (danger nearby,
  heading, food direction). It learns quickly: after a few hundred games
  the snake eats **~15 food per game** on an 8×8 board.
* **`--net grid`** feeds it the raw board as a small image and grows a
  couple of convolutions -- the *same* DQN, now with eyes. This is the
  honest, harder task, and convolution in pure NumPy/CuPy is heavy, so it
  learns slowly: expect a snake that reliably avoids walls and starts
  nibbling food, not a champion. It is here to show that BabyTorch's
  `Conv2D` drops straight into an RL loop -- reward shaping (`--shaping`)
  helps its sparse signal a lot.

## Where the ideas come from

The book's **Part III** ([chapters 9–10](../../book/README.md)) explains
all of this from the ground up -- the RL problem, the policy gradient, the
Bellman equation, and PPO -- using the exact code in these files. The
classic tabular precursors (Value Iteration, SARSA, Q-learning) live in
the [original repository](https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch).
