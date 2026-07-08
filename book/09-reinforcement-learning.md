# Chapter 9 — Reinforcement Learning

Every chapter so far learned from a **dataset**: a pile of examples with
answers attached. A GPT learns from text where the "answer" is the next
character; a classifier learns from images with labels. The loss compared
the model's guess to a known truth.

Reinforcement learning throws the answer key away. There is no dataset.
An **agent** is dropped into a game, it takes **actions**, and all the
game ever tells it is a **reward** -- a number, good or bad. Nobody says
which action was right. The agent has to *discover* good behaviour by
trying things and noticing what pays off.

That is a genuinely different kind of learning, and it needs only one new
idea on top of Part I. Everything else -- `Linear`, `Adam`,
`log_softmax`, `backward()` -- you already have. In fact this whole
chapter is a port of a separate PyTorch project,
[deep-reinforcement-learning-games-from-scratch](https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch),
onto BabyTorch -- and the port is nearly a rename, because BabyTorch
mirrors PyTorch. That is the subtext of Part III: if you can build a
classifier, you can build an agent.

## The loop: agent, environment, reward

![The reinforcement-learning loop: an agent holding a policy pi(a given s) sends an action to the environment; the environment answers with a reward and the next state, which flow back to the agent; the agent's goal is to choose actions that maximise the discounted return](figures/fig-rl-loop.svg)

The whole of RL is that loop. At each step the agent sees a **state**
`s` (what the game looks like now), picks an **action** `a`, and the
environment answers with a **reward** `r` and the next state `s'`. Round
and round.

The agent's goal is not to maximise the *next* reward -- it is to maximise
the **return**, the total reward over the whole episode, with later
rewards discounted a little:

```
G = r₀ + γ r₁ + γ² r₂ + …            (0 < γ < 1)
```

The discount `γ` (gamma, ~0.99) means "a reward now is worth slightly more
than the same reward later" -- which, usefully, also makes *shorter* paths
to a reward worth more than long ones.

Our first game is **GridWorld**: an agent on a small grid, walls to
avoid, a goal to reach. Every step costs `-1`; reaching the goal pays a
`+10` bonus and ends the episode. So the return is high only for a short
successful path -- exactly the behaviour we want the agent to find. It
speaks the same `reset` / `step` language as any Gym environment:

```python
from gridworld import GridWorld
env = GridWorld()
obs, _ = env.reset()                       # obs: a 6-number view of the cell
obs, reward, done, info = env.step(2)      # action in 0..3 (up/down/left/right)
```

## The policy: a network that chooses

The agent's brain is a **policy**: a network that reads the state and
outputs one score (logit) per action. Softmax turns those into
probabilities, and the agent *samples* an action from them -- sampling,
not argmax, because an agent that never tries anything new can never
discover that it was wrong.

That sample-and-score step is the one piece PyTorch hands you
(`torch.distributions.Categorical`) that BabyTorch does not. So we write
it -- in a dozen lines, entirely out of `log_softmax` and the fancy
indexing from chapter 3's cross-entropy:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/common.py</code> (the categorical distribution)</summary>

```python
class Categorical:
    # ...
    def __init__(self, logits):
        self.logits = logits                     # (B, n_actions) Tensor
        self.log_probs = logits.log_softmax(axis=-1)   # (B, n_actions)

    def sample(self):
        """Return one sampled action id per row, as a NumPy int array."""
        probs = np.exp(to_numpy(self.log_probs.data))       # (B, n)
        # Inverse-CDF sampling, vectorised over the batch: draw u ~ U(0,1)
        # and take the first action whose cumulative probability exceeds it.
        u = np.random.random((probs.shape[0], 1))
        return (probs.cumsum(axis=1) > u).argmax(axis=1)

    def log_prob(self, actions):
        """Log-probabilities of ``actions`` (a 1-D int array) -> Tensor (B,)."""
        actions = xp.asarray(actions).astype(xp.int64)
        return self.log_probs[xp.arange(actions.shape[0]), actions]

    def entropy(self):
        """Entropy of each row, ``-sum(p * log p)`` -> Tensor (B,)."""
        return -(self.log_probs.exp() * self.log_probs).sum(axis=-1)
```

</details>

`log_prob` is the important one: it returns a Tensor still wired into the
graph, so `backward()` can flow gradients into the policy through the
action it chose. Hold onto that -- it *is* the policy gradient.

## REINFORCE: make good actions more likely

Here is the entire idea of policy gradients, in one sentence: **play an
episode, and for every action taken, push its probability up if the
return was good and down if it was bad.**

![Policy gradient turns one played episode into a better policy: a trajectory of states, actions and rewards becomes a return G-t for each step, which weights the loss minus log pi(a-t given s-t) times the advantage A-t; a side panel shows REINFORCE, Actor-Critic and PPO differ only in how they compute the advantage](figures/fig-policy-gradient.svg)

The loss for one action is `-log π(a|s) · Aₜ`, where `Aₜ` is the
**advantage** -- how good the outcome was, relative to a baseline.
Minimising it (gradient descent) *raises* the log-probability of actions
with positive advantage. That minus sign is the whole trick: it turns
"maximise reward" into a loss `backward()` can chew on.

REINFORCE uses the simplest baseline there is -- the average return over a
batch of episodes -- so `Aₜ` just asks "did this action beat a typical
one?". The update is four lines:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/reinforce.py</code> (the policy-gradient step)</summary>

```python
        states = np.concatenate(b_states)
        actions = np.concatenate(b_actions)
        advantages = C.to_tensor(C.normalize(np.concatenate(b_returns)))

        # --- one policy-gradient step --------------------------------------
        dist = C.Categorical(policy(C.to_tensor(states)))
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * advantages).mean() - args.ent_coef * dist.entropy().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

</details>

The `entropy` term is a gentle bribe to keep exploring: it rewards the
policy for staying a little uncertain, so it doesn't collapse onto one
action before it has really learned. Run it and the agent climbs from
wandering (return ≈ −100) to solving the maze every time (return ≈ +2) in
under a hundred updates.

## Actor-Critic: learn the baseline

REINFORCE's baseline is a single number for the whole batch. But a state
near the goal is *genuinely* better than one in a far corner, and a flat
baseline can't tell them apart -- so the advantage stays noisy, and
learning is slower than it needs to be.

The fix is a second network, the **critic** `V(s)`, trained to predict the
return from each state. Now the advantage is `return − V(s)`: "how much
better than expected was this?", judged per state. Two networks, two
losses, one shared rollout:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/actor_critic.py</code> (advantage, then two updates)</summary>

```python
        values = critic(states)                              # V(s), with grad
        # Advantage is a *fixed weight* for the actor, so compute it in NumPy
        # (this is the "detach the critic" step): how much each return beat
        # the critic's expectation, then normalised for a steady gradient.
        advantages = C.to_tensor(C.normalize(returns - to_numpy(values.data)))

        # --- actor update: policy gradient, weighted by advantage ----------
        dist = C.Categorical(actor(states))
        actor_loss = (-(dist.log_prob(actions) * advantages).mean()
                      - ent_coef * dist.entropy().mean())
        opt_actor.zero_grad()
        actor_loss.backward()
        opt_actor.step()

        # --- critic update: regress V(s) toward the actual returns ---------
        critic_loss = ((C.to_tensor(returns) - values) ** 2).mean()
        opt_critic.zero_grad()
        critic_loss.backward()
        opt_critic.step()
```

</details>

The critic is trained by plain regression -- MSE against the returns it is
trying to predict, exactly the loss from chapter 3. The actor uses the
critic's verdict as a smarter baseline. Same policy gradient; less noise.

## PPO: the one everyone actually uses

Actor-Critic works but is wasteful and fragile: each batch of hard-won
experience powers a single gradient step and is thrown away, and nothing
stops that step from being so large it destroys the policy.

**Proximal Policy Optimization** fixes both, and is the workhorse behind
most modern RL -- including the RLHF that aligns language models (the
"reward" there is a human-preference model, but the algorithm is this
one). Its trick is the **clipped objective**. Track the ratio between the
new policy and the one that collected the data, `r = π_new / π_old`, and
clip it to `[1−ε, 1+ε]` so a single update can never move the policy too
far:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/ppo.py</code> (the clipped surrogate objective)</summary>

```python
            dist = C.Categorical(actor(s))
            new_logp = dist.log_prob(actions[idx])
            # ratio = pi_new / pi_old, computed in log-space for stability.
            ratio = (new_logp - C.to_tensor(old_logps[idx])).exp()
            adv = C.to_tensor(advantages[idx])

            # The clipped surrogate: take the *pessimistic* of the raw and
            # clipped objectives, so an update that would move the policy
            # outside [1-eps, 1+eps] earns no extra reward.
            unclipped = ratio * adv
            clipped = C.clip_tensor(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -C.minimum(unclipped, clipped).mean()
```

</details>

That safety rail is what lets PPO **reuse each batch for several epochs**
of minibatch updates instead of one -- far more sample-efficient. (The
`clip_tensor` and `minimum` are elementwise, differentiable, and built
from `relu` -- BabyTorch's `Tensor.max` only reduces along an axis, so
[`common.py`](../tutorials/rl/common.py) makes the elementwise versions
the same way chapter 2 built everything else.) On GridWorld it is the
runaway winner: solved in ~15 updates, where REINFORCE needs ~70.

## Run it

```bash
cd tutorials/rl
python reinforce.py        # policy gradient
python actor_critic.py     # + a learned baseline
python ppo.py              # clipped, sample-efficient
python reinforce.py --render_every 20     # watch it think, in ASCII
```

Each prints its progress, saves a learning curve, and shares one lesson:
an "agent" is a network trained by a slightly unusual loss. The
[tutorial README](../tutorials/rl/README.md) has the full table; the next
chapter takes the *other* road -- learning values instead of a policy --
and uses it to play Snake.

## Exercises

**Check yourself** (answers unfold):

**Q1.** REINFORCE's loss is `-(log_prob * advantage)`. If an action led
to a *below-average* return (negative advantage), which way does the
update move that action's probability, and why?

<details><summary>Answer</summary>

Down. With a negative advantage, `-(log_prob · advantage)` is *minimised*
by making `log_prob` **smaller** (less negative overall), so gradient
descent lowers that action's probability. Good-outcome actions (positive
advantage) get pushed up, bad ones down -- exactly the intent.

</details>

**Q2.** Why sample from the policy during training instead of always
taking the most likely action (argmax)?

<details><summary>Answer</summary>

Exploration. An agent that always takes its current-best action never
tries the alternatives, so it can never discover that a different action
was actually better -- it gets stuck. Sampling (plus the entropy bonus)
keeps it trying things. At *evaluation* time, once learning is done, you
switch to greedy argmax.

</details>

**Q3.** Actor-Critic and PPO both compute an advantage. What breaks if
you feed the *critic's* gradient into the actor's loss (i.e. forget to
treat the advantage as a constant)?

<details><summary>Answer</summary>

The advantage is meant to be a fixed *weight* on the policy gradient, not
something to optimise. If gradients flow through it into the critic, the
actor update starts nudging the critic to make advantages look good rather
than to predict returns -- the baseline stops being an honest baseline and
training destabilises. That is why the code computes the advantage in
NumPy (or would `.detach()` it): a deliberate stop-gradient.

</details>

**Build it** — the value-based agent in the next chapter needs a Huber
loss and a replay buffer; try sketching `discounted_returns` and
`compute_gae` from their docstrings in
[`tutorials/rl/common.py`](../tutorials/rl/common.py) before you read them.

---

**The code:**
[`tutorials/rl/common.py`](../tutorials/rl/common.py) ·
[`tutorials/rl/gridworld.py`](../tutorials/rl/gridworld.py) ·
[`tutorials/rl/reinforce.py`](../tutorials/rl/reinforce.py) ·
[`tutorials/rl/actor_critic.py`](../tutorials/rl/actor_critic.py) ·
[`tutorials/rl/ppo.py`](../tutorials/rl/ppo.py)

[← Chapter 8: Training a GPT](08-training-a-gpt.md) | [Contents](README.md) | [Chapter 10: Deep Q-Learning →](10-deep-q-learning.md)
