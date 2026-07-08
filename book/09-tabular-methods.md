# Chapter 9 — Tabular Reinforcement Learning

Every chapter so far learned from a **dataset**: a pile of examples with
answers attached. A GPT learns from text where the "answer" is the next
character; a classifier learns from images with labels. The loss compared
the model's guess to a known truth.

Reinforcement learning throws the answer key away. There is no dataset.
An **agent** is dropped into a game, it takes **actions**, and all the
game ever tells it is a **reward** -- a number, good or bad. Nobody says
which action was right. The agent has to *discover* good behaviour by
trying things and noticing what pays off.

That is a genuinely different kind of learning, and Part III builds it in
two moves. This chapter makes the *first* move on the smallest possible
stage: a grid you can hold in your head, where the agent's knowledge fits
in a plain **table** of numbers -- one per state -- and not a single
neural network is in sight. Get the ideas clear here, with nothing to
hide behind, and the next chapter's move is easy: swap the table for a
network, and tabular RL becomes *deep* RL.

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

The game we learn on is **GridWorld**: an agent on a small grid, walls
to avoid, a goal to reach. Every step costs `-1`; reaching the goal pays a
`+10` bonus and ends the episode. So the return is high only for a short
successful path -- exactly the behaviour we want the agent to find. It
speaks the same `reset` / `step` language as any Gym environment:

```python
from gridworld import GridWorld
env = GridWorld()
obs, _ = env.reset()                       # obs: a 6-number view of the cell
obs, reward, done, info = env.step(2)      # action in 0..3 (up/down/left/right)
```

Because GridWorld is a 5×5 board with a wall down the middle, it has only
a couple of dozen cells. That is small enough to name **every state** --
and that is exactly what makes a table possible.

## The value of a state

Give up on choosing actions for a moment and ask a simpler question: how
*good* is it to be in a given cell? Call that number the state's **value**
`V(s)` -- the best return you can still collect starting from `s`. A cell
next to the goal is worth almost `+9`; a cell in the far corner is worth
much less, because it costs a long walk of `-1`s to get out.

Values have a beautiful self-referential structure. The value of a cell is
the reward for stepping out of it plus the (discounted) value of wherever
that step lands -- taking the *best* step available:

```
V(s)  =  maxₐ [ r(s, a)  +  γ · V(s') ]
```

This is the **Bellman equation**, and it is the engine of everything in
this chapter and the next. It is a fixed point: the true values are the
ones that make both sides equal. Store one number per state in a
dictionary -- the **table** -- and the job is to fill it with values that
satisfy the equation. Closely related is the **action-value** `Q(s, a)`:
the value of taking action `a` *first*, then acting well. Once you have
either table, acting is trivial -- in each state, take the action with the
highest value.

Two ways to fill the table. If you are *handed the rules of the game* --
every state, and where each action leads -- you can compute the values
directly; that is **dynamic programming**. If you are not, you have to
*play* and learn from what happens; that is **temporal-difference
learning**. GridWorld lets us do both.

## Model-based: dynamic programming

Dynamic programming is the easy case: the agent is given a **model** of
the world. GridWorld hands it three methods -- `states()` (every cell),
`is_terminal(s)` (is this the goal?), and `transition(s, a)` (where does
this action lead, and for what reward?). Because GridWorld is
deterministic, a transition is a single `(s', r)` pair, not a probability
distribution -- which keeps the Bellman backup a plain `max`, with no
expectation to average over.

**Value iteration** applies the Bellman equation as an *assignment*: sweep
every state, overwrite its value with the best `r + γ · V(s')`, and repeat
until nothing changes. Each sweep pushes value one step further out from
the goal; after enough sweeps the whole board is correct.

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/tabular.py</code> (the Bellman optimality backup)</summary>

```python
def value_iteration(env, gamma=0.99, theta=1e-6):
    # ...
    V = {s: 0.0 for s in env.states()}
    while True:
        delta = 0.0
        for s in env.states():
            if env.is_terminal(s):
                continue
            best = max(r + gamma * V[ns]
                       for ns, r in (env.transition(s, a)
                                     for a in range(env.n_actions)))
            delta = max(delta, abs(best - V[s]))
            V[s] = best
        if delta < theta:
            break
    return V, _greedy_from_v(env, V, gamma)
```

</details>

The loop stops when the largest change in any cell (`delta`) falls below a
tiny `theta` -- the table has stopped moving, so it has converged. The
returned policy is *greedy*: in each state, pick the action whose
`r + γ · V(s')` is largest.

**Policy iteration** reaches the same optimum by a different route. It
alternates two exact steps: *evaluate* the current policy (solve `V` for
the actions it currently takes, no `max`), then *improve* it (act greedily
with respect to that `V`). Repeat until the policy stops changing. It
often takes fewer sweeps than value iteration, because each evaluation
solves the current policy exactly instead of taking one Bellman step. Both
land on the identical optimal policy -- there is only one best way through
the maze.

## Model-free: learning by playing

Dynamic programming needs the rulebook. But an agent facing a real game
does *not* get a `transition` function -- it only gets to move and see
what happens. **Temporal-difference (TD) control** learns from exactly
that: play the game, and after each step nudge the value of where you
*were* toward the reward you got plus the value of where you *landed*.
That "plus the value of where you landed" is the Bellman equation again,
used as a learning target instead of an assignment.

TD learns a `Q` table (action-values), because without a model you cannot
turn a state-value into an action -- you need the value of the actions
themselves. The state here is simply the agent's cell, the `(row, col)`
tuple `env.pos`, so `Q` is a dictionary from cell to four action-values.
To keep exploring, the agent acts **ε-greedy**: usually take
the best-known action, but with probability ε pick at random, so it never
stops discovering. That single helper is the source of all exploration:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/tabular.py</code> (ε-greedy action choice)</summary>

```python
def _epsilon_greedy(Q, s, n_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[s]))
```

</details>

**SARSA** is the *on-policy* version: it updates toward the value of the
action it will *actually take next* (`a'`, drawn from the same ε-greedy
policy). So it learns the value of the behaviour it truly follows,
exploration and all. The name is the transition it uses -- state, action,
reward, state, action:

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/tabular.py</code> (SARSA: update toward the action actually taken)</summary>

```python
        while not done:
            _, r, done, _ = env.step(a)
            ns = env.pos
            na = _epsilon_greedy(Q, ns, env.n_actions, epsilon)
            target = r + (0.0 if done else gamma * Q[ns][na])
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = ns, na
```

</details>

**Q-learning** is the *off-policy* version, and the one change is the
whole point: the target uses `max Q(s', ·)` -- the value of the *best*
next action -- rather than the value of the action it happens to take. So
it learns the optimal policy's values while still wandering ε-greedily to
gather experience. It is the direct tabular ancestor of the DQN in the
next chapter.

<details>
<summary><b>How it's implemented</b> — <code>tutorials/rl/tabular.py</code> (Q-learning: update toward the best next action)</summary>

```python
        while not done:
            a = _epsilon_greedy(Q, s, env.n_actions, epsilon)
            _, r, done, _ = env.step(a)
            ns = env.pos
            target = r + (0.0 if done else gamma * np.max(Q[ns]))
            Q[s][a] += alpha * (target - Q[s][a])
            s = ns
```

</details>

The `target − Q[s][a]` is the **TD error**: the gap between what the agent
now believes and what it believed a step ago. `alpha` (the learning rate)
controls how far each step closes that gap. No gradients, no `backward()`
-- just a running average, drifting toward the Bellman fixed point one
played step at a time.

## Run it

```bash
cd tutorials/rl
python tabular.py
```

All four methods print their policy as arrows on the board. The two
model-based methods agree exactly -- one optimal path -- and the model-free
ones recover it too, at least on the cells that matter:

```
Value iteration (model-based, exact):
↓ → ↓ ↓ ↓
↓ # ↓ ↓ ↓
↓ # ↓ ↓ ↓
↓ # ↓ ↓ ↓
→ → → → G

Q-learning (model-free, off-policy, 500 episodes):
→ → ↓ ↓ ↓
↑ # ↓ ↓ ↓
↓ # → → ↓
↓ # ↓ ↓ ↓
→ → → → G

Q-learning agrees with the exact optimum on 81% of states.
```

The disagreements are all on cells *off* the good paths -- corners the
optimal agent never visits, so Q-learning barely sampled them and never
sharpened their values. On every cell that lies on a sensible route to the
goal, the played-out table matches the exact one. That is the model-free
promise: no rulebook, and it still finds the way.

## Why a table isn't enough

Everything here rests on one luxury: a table with **one cell per state**.
GridWorld has a couple of dozen. But make the grid a photograph, the board
a game of Go, or the state a paragraph of text, and the number of states
explodes past every atom in the universe -- no table could hold them, and
the agent would never visit any single state twice to learn it.

The fix is the idea the rest of Part III is built on: **replace the table
with a network**. A network reads the state and *generalises* -- it gives
a sensible value for states it has never seen, the same way chapter 3's
classifier labels images it was never trained on. Chapter 10 does exactly
this to the *policy* (learn how to act directly), and chapter 11 does it
to this chapter's `Q` table (learn what actions are worth). Every idea you
just met -- return, discount, Bellman target, greedy action, exploration
-- carries straight over. Only the table changes.

## Exercises

**Check yourself** (answers unfold):

**Q1.** Value iteration needs `env.transition` and `env.states`;
Q-learning needs neither. What can Q-learning do that value iteration
cannot -- and what does it give up for that?

<details><summary>Answer</summary>

Q-learning can learn in a world whose rules it does not know -- it only has
to be able to *play* it, which is the realistic case (no one hands you a
transition function for Atari or Go). What it gives up is exactness and
efficiency: it only learns about states it actually visits, so rarely-seen
cells stay wrong (the 81%), and it needs many episodes of trial and error
where dynamic programming computes the answer in a few sweeps.

</details>

**Q2.** SARSA and Q-learning differ in a single term: SARSA's target uses
`Q[ns][na]` (the next action actually chosen), Q-learning's uses
`max Q[ns]` (the best next action). Which one learns the value of a
*safe* path when exploration is risky, and why?

<details><summary>Answer</summary>

SARSA. Because it bootstraps off the action it *will actually take* --
including the occasional random ε-greedy move -- its values account for the
cost of exploring. If wandering near a cliff sometimes falls off, SARSA
learns to give the cliff edge a wide berth. Q-learning learns the value of
the *optimal* policy regardless of the exploratory moves it makes, so it
will happily walk the cliff edge, trusting it would act perfectly. On
GridWorld, where a wrong step is cheap, both converge to the same route.

</details>

**Q3.** In `q_learning`, the target is `r + (0.0 if done else gamma *
np.max(Q[ns]))`. Why is the future term forced to zero on the step that
reaches the goal?

<details><summary>Answer</summary>

Because nothing follows a terminal state -- the episode is over, so there
is no next action and no future reward to add. Bootstrapping off `Q[ns]`
there would invent value out of thin air (and `ns` is the absorbing goal,
whose value should be exactly its own reward). Zeroing the future term
anchors the whole table: it is the one place a value is known *without*
reference to another value, and every other cell's value is ultimately
propagated back from it.

</details>

**Build it** — add a `render_values` helper that prints `V(s)` for every
cell as a number grid (the way `render_policy` prints arrows), and watch
value spread outward from the goal one sweep at a time by calling it
inside `value_iteration`'s loop.

---

**The code:**
[`tutorials/rl/tabular.py`](../tutorials/rl/tabular.py) ·
[`tutorials/rl/gridworld.py`](../tutorials/rl/gridworld.py) ·
[`tests/test_rl.py`](../tests/test_rl.py) (the four methods, proven to agree)

[← Chapter 8: Training a GPT](08-training-a-gpt.md) | [Contents](README.md) | [Chapter 10: Policy Gradients →](10-policy-gradients.md)
