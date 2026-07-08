# Chapter 10 — Deep Q-Learning

Chapter 9 learned a **policy** directly: a network that maps a state to
action probabilities. This chapter takes reinforcement learning's other
great road. Instead of learning *how to act*, learn *what things are
worth* -- and then act greedily on that.

The thing to learn is the **Q-function**, `Q(s, a)`: the total future
reward you can expect if you take action `a` in state `s` and play well
afterward. Once you have it, the policy is free -- in any state, take the
action with the highest Q. No policy network at all.

## The Bellman equation

Q has a beautiful self-referential structure. The value of an action is
its immediate reward plus the discounted value of the *best* action
available next:

```
Q(s, a)  =  r  +  γ · maxₐ′ Q(s′, a′)
```

This is the **Bellman equation**, and it is the entire training signal.
It is a fixed point: a correct Q makes both sides equal. So we train a
network to make its left side match its own (bootstrapped) right side.
Classical RL fills in a `Q` *table*, one cell per state-action pair (the
tabular [Q-learning](https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch/tree/main/01_Reinforcement_learning/03_Qlearning)
in the original repo does exactly that). **Deep Q-Learning** replaces the
table with a network, so it can generalise across states it has never
seen -- essential the moment the state is an image.

## DQN: two tricks that make it stable

Training a network to chase its own output is a recipe for oscillation.
DQN (Mnih et al., 2015 -- the Atari paper) tames it with two ideas.

![DQN: the environment feeds transitions into a replay buffer; a random batch is drawn and sent to two networks; the Q-network scores Q(s,a) and feeds the Huber loss, while a slow target network computes the Bellman target r plus gamma times the max next-state Q, which also feeds the loss; the loss's gradient trains the Q-network, and a soft update slowly copies it into the target network](figures/fig-dqn.svg)

* **Experience replay** -- store every transition `(s, a, r, s′)` in a
  buffer and train on *random* batches from it. Consecutive steps are far
  too correlated to learn from directly; sampling the buffer shuffles that
  correlation away.
* **A target network** -- a slowly-trailing copy of the Q-network supplies
  the `maxₐ′ Q(s′, a′)` on the right-hand side. If the same fast-moving
  network sat on both sides, the target would lurch every step and the
  chase would never settle.

Exploration is **ε-greedy**: act randomly with probability ε (high at
first, decaying as the agent learns), and take the argmax of Q otherwise.
The core update is the Bellman equation, made into a loss:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/dqn.py</code> (one DQN update)</summary>

```python
        s, a, r, s2, done = memory.sample(batch_size)

        # Q(s, a): the network's current estimate for the actions taken.
        q_sa = q_net(C.to_tensor(s))[xp.arange(batch_size), xp.asarray(a)]

        # The Bellman target r + gamma * max_a' Q_target(s'), zeroed past a
        # terminal state (nothing follows the goal).  It is a fixed target,
        # so build it with no_grad and feed it in as a constant.
        with babytorch.no_grad():
            next_q = to_numpy(target_net(C.to_tensor(s2)).max(axis=1).data)
        target = C.to_tensor(r + gamma * (1.0 - done) * next_q)

        loss = C.huber_loss(q_sa, target)
        optimizer.zero_grad()
        loss.backward()
        C.clip_grad_value(q_net.parameters(), grad_clip)
        optimizer.step()
```

</details>

Two BabyTorch details worth noticing. `q_net(...)[xp.arange(batch_size),
xp.asarray(a)]` is the same fancy-indexing **gather** that cross-entropy
used in chapter 3 -- here it picks out `Q(s, a)` for the action actually
taken. And the target is built inside `no_grad` and re-wrapped as a
constant: gradients must flow through `Q(s, a)`, never through the target,
or the network would chase a moving copy of itself.

### A loss built for moving targets

Bootstrapped targets are noisy, and a single wild one under plain MSE
produces a giant gradient that can wreck the network. DQN uses the
**Huber loss** instead -- squared for small errors, linear for large ones
-- and, like every loss in this book, it is built from primitives, so
`backward()` differentiates it for free:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/common.py</code> (Huber loss from relu)</summary>

```python
def huber_loss(pred, target, delta=1.0):
    # ...
    err = pred - target
    abs_err = err.relu() + (-err).relu()                 # |err|
    quadratic = 0.5 * err * err
    linear = delta * (abs_err - 0.5 * delta)
    # Pick the quadratic branch where |err| <= delta, the linear one beyond.
    small = Tensor((abs_err.data <= delta).astype(xp.float32))
    return (small * quadratic + (1.0 - small) * linear).mean()
```

</details>

`err.relu() + (-err).relu()` is `|err|` -- absolute value out of two ReLUs,
in the spirit of chapter 2. On GridWorld the result is a Q-function whose
greedy policy solves the maze **100%** of the time, even while it was
still exploring with random moves during training.

## Snake: the same agent, with eyes

Nothing about DQN mentions GridWorld. Point it at a harder game and it
just works -- so we point it at **Snake**. The agent is literally
`dqn.py`; only the game and the network's *eyes* change.

The first eyes are hand-built: an 11-number summary of the board -- is
there danger straight / left / right, which way am I heading, where is the
food. A tiny MLP reads that and, after a few hundred games, drives a snake
that eats **~15 food per game**:

```bash
cd tutorials/rl
python snake_dqn.py --net features
```

The honest, harder version gives the agent the **raw board** as a small
image and lets it learn its own features with convolutions -- the same
DQN, now with a `ConvNet` where the MLP was:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/common.py</code> (a convolutional Q-network)</summary>

```python
class ConvNet(nn.Module):
    # ...
    def __init__(self, input_shape, n_actions, hidden=128):
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2D(c, 16, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2D(16, 32, kernel_size=3, stride=1), nn.ReLU())
        self.flatten = nn.Flatten()
        # Discover the flattened size by pushing one zero image through.
        with babytorch.no_grad():
            n_flat = self.flatten(self.conv(babytorch.zeros(1, c, h, w))).shape[1]
        self.head = mlp([n_flat, hidden, n_actions])

    def forward(self, x):
        return self.head(self.flatten(self.conv(x)))
```

</details>

```bash
python snake_dqn.py --net grid --rows 6 --cols 6 --shaping 0.5
```

This is where "baby" earns its name. Convolution in pure NumPy/CuPy is
slow, and learning to see a whole board from scratch takes far more
experience than reading eleven tidy features -- so the pixel agent learns
slowly, and this demo shows a snake that learns to avoid walls and starts
nibbling, not a champion. But it runs the identical training loop, and it
proves the point of the whole framework: chapter 3's `Conv2D` drops
straight into an RL agent with nothing new required.

## Where reinforcement learning goes

Between this and a game-playing or chat-aligning system at the frontier
lie scale and a handful of refinements -- Double DQN and duelling heads,
prioritised replay, stacking several frames so the agent can see motion,
and thousands of parallel actors. But the ideas are the ones in these two
chapters.

And they close a loop with Part II. The "PO" in the RLHF that turns a raw
language model into a helpful assistant is the **PPO** of chapter 9: the
policy is the LLM, an action is the next token, and the reward comes from
a model trained on human preferences. The Transformer you built and the
agent you just trained are the same machine, pointed at different games.

## Exercises

**Check yourself** (answers unfold):

**Q1.** In the Bellman target `r + γ · maxₐ′ Q_target(s′, a′)`, why use
the *target* network for the max instead of the live Q-network?

<details><summary>Answer</summary>

Stability. The target is what the live network is being trained to match.
If the live network appeared on *both* sides, every update would change
the target it is chasing, and the two would oscillate. A slow-moving copy
keeps the target roughly fixed for many steps, so the regression has
something stable to converge to.

</details>

**Q2.** During DQN training on GridWorld the average return is noisy and
often negative, yet the *greedy* evaluation solves the maze every time.
How can both be true?

<details><summary>Answer</summary>

The training episodes still take random (ε-greedy) actions, and a single
random move can send the snake into a wall or the agent the long way
round -- so the *behaviour* policy scores poorly. The greedy policy turns
exploration off and follows the learned Q, which is already correct. That
is exactly why the value-based agents are judged by a separate greedy
evaluation.

</details>

**Q3.** The Snake feature agent eats ~15 food per game after a few
hundred episodes; the pixel (ConvNet) agent barely nibbles after far more.
Same algorithm -- what makes the pixel version so much harder?

<details><summary>Answer</summary>

The features hand the agent the answer -- danger directions and where the
food is -- so it only has to learn *what to do*. The pixel agent must
*first* learn to see: to extract "there is a wall ahead" and "food is up
and to the left" from a grid of numbers, using convolutions it is
training from scratch. That is a far bigger learning problem, and it needs
much more experience (and, in pure NumPy, much more patience).

</details>

**Build it** — add **Double DQN**: use the live network to *choose* the
next action but the target network to *value* it (it curbs DQN's tendency
to overestimate). It is a two-line change to the target computation in
[`tutorials/rl/dqn.py`](../tutorials/rl/dqn.py).

---

**The code:**
[`tutorials/rl/dqn.py`](../tutorials/rl/dqn.py) ·
[`tutorials/rl/snake.py`](../tutorials/rl/snake.py) ·
[`tutorials/rl/snake_dqn.py`](../tutorials/rl/snake_dqn.py) ·
[`tutorials/rl/common.py`](../tutorials/rl/common.py) ·
[`tests/test_rl.py`](../tests/test_rl.py) (every agent, proven to learn)

[← Chapter 9: Reinforcement Learning](09-reinforcement-learning.md) | [Contents](README.md) | *End of the book — [back to the repository](../README.md)*
