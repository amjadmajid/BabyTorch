"""DQN: learn the *value* of actions, and act greedily on it.

The policy-gradient methods learn to act directly. DQN takes the other
great road in RL: learn a **Q-function** ``Q(s, a)`` -- the total future
reward you can expect from taking action ``a`` in state ``s`` -- and then
just pick the action with the highest Q. No policy network at all; the
policy is "argmax over Q".

Q obeys the **Bellman equation**: the value of an action is its immediate
reward plus the (discounted) value of the best action next --

    Q(s, a)  ->  r + gamma * max_a' Q(s', a')

So the network is trained to make its left side match its own right side.
Two famous tricks stop that self-reference from exploding:

* **Experience replay** -- learn from random past transitions, not just
  the latest step, so the training batches aren't hopelessly correlated;
* **a target network** -- a slowly-trailing copy of Q supplies the
  ``max_a' Q(s', a')`` on the right, so the target doesn't lurch every
  step.

Exploration is **epsilon-greedy**: act randomly with probability epsilon
(high at first, decaying), act greedily otherwise.

Ported from the DQN agent in
https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch

    python dqn.py
    python dqn.py --episodes 400 --plot dqn.png
    BABYTORCH_DEVICE=cpu python dqn.py
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import babytorch
from babytorch.backend import xp, to_numpy
from babytorch.optim import Adam

import common as C
from gridworld import GridWorld


def epsilon(step, start=0.9, end=0.05, decay=2000):
    """Exploration rate: starts high, decays smoothly toward ``end``."""
    return end + (start - end) * math.exp(-step / decay)


def train(episodes=400, gamma=0.99, lr=1e-3, hidden=128, batch_size=64,
          capacity=10000, tau=0.01, grad_clip=10.0, warmup=200,
          eps_start=0.9, eps_end=0.02, eps_decay=1200,
          seed=0, env=None, net_fn=None, on_episode=None):
    C.set_seed(seed)
    env = env or GridWorld()
    # By default the Q-network is a small MLP over the observation vector;
    # pass ``net_fn`` to use, say, a ConvNet for image observations (Snake).
    net_fn = net_fn or (lambda e: C.QNet(e.obs_dim, e.n_actions, hidden))

    q_net, target_net = net_fn(env), net_fn(env)
    C.hard_update(target_net, q_net)                 # start them identical
    optimizer = Adam(q_net.parameters(), learning_rate=lr)
    memory = C.ReplayBuffer(capacity)

    def select_action(obs, eps):
        """Epsilon-greedy: explore at random, else act on the best Q."""
        if np.random.random() < eps:
            return np.random.randint(env.n_actions)
        with babytorch.no_grad():
            q = q_net(C.to_tensor(obs[None]))         # add a batch axis
        return int(q.argmax(axis=1)[0])

    def optimize():
        if len(memory) < max(batch_size, warmup):
            return
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

    returns_history = []
    total_steps = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return, done = 0.0, False
        while not done:
            eps = epsilon(total_steps, eps_start, eps_end, eps_decay)
            action = select_action(obs, eps)
            next_obs, reward, done, info = env.step(action)
            memory.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_return += reward
            total_steps += 1

            optimize()
            # Nudge the target network a hair toward the live one each step.
            C.soft_update(target_net, q_net, tau)

        returns_history.append(ep_return)
        if on_episode:
            eps = epsilon(total_steps, eps_start, eps_end, eps_decay)
            on_episode(ep, returns_history, eps)
    return q_net, returns_history


def main():
    p = argparse.ArgumentParser(description="DQN on GridWorld.")
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default="dqn.png")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "gpu"])
    args = p.parse_args()

    if args.device:
        babytorch.set_device(args.device)
    print(f"Device: {babytorch.device()}\n")

    def report(ep, history, eps):
        if ep % 25 == 0 or ep == args.episodes - 1:
            recent = history[-25:]
            print(f"episode {ep:4d} | avg return {np.mean(recent):7.2f} | "
                  f"solved {100 * np.mean([r > 0 for r in recent]):5.1f}% | "
                  f"epsilon {eps:.2f}")

    q_net, history = train(
        episodes=args.episodes, gamma=args.gamma, lr=args.lr, hidden=args.hidden,
        batch_size=args.batch_size, tau=args.tau, seed=args.seed, on_episode=report)

    # The real measure of a Q-function: how good is its *greedy* policy,
    # once we stop exploring?
    avg_return, solved = C.evaluate(GridWorld(), q_net, episodes=50)
    print(f"\nGreedy evaluation over 50 episodes: "
          f"avg return {avg_return:.2f}, solved {100 * solved:.0f}%")
    C.plot_returns(history, args.plot, "DQN on GridWorld", window=25)


if __name__ == "__main__":
    main()
