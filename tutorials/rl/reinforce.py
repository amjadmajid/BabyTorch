"""REINFORCE: the policy-gradient idea in its purest form.

Train a network to *act*, directly, with no value function and no
labelled data. The recipe is astonishingly short:

1. Play a whole episode by sampling actions from the policy.
2. For each step, compute its **return** -- the total reward that
   followed it.
3. Nudge the policy to make the actions that led to high returns more
   likely, and the ones that led to low returns less likely.

That third step is one line: ``loss = -(log_prob(action) * return)``.
Minimising it maximises return-weighted log-probability -- gradient
*ascent* on reward, dressed as gradient descent on a loss. BabyTorch's
``backward()`` does the rest.

Ported from the REINFORCE agent in
https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch

    python reinforce.py                 # train on GridWorld
    python reinforce.py --episodes 1000 --render_every 200
    BABYTORCH_DEVICE=cpu python reinforce.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import babytorch
from babytorch.optim import Adam

import common as C
from gridworld import GridWorld


def run_episode(env, policy, render=False):
    """Play one episode; return the states, actions and rewards seen."""
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    done = False
    # Acting needs no gradients -- we only record what happened.
    with babytorch.no_grad():
        while not done:
            if render:
                env.render()
            logits = policy(C.to_tensor(obs).reshape(1, -1))
            action = int(C.Categorical(logits).sample()[0])
            next_obs, reward, done, info = env.step(action)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
    return np.array(states, dtype=np.float32), np.array(actions), rewards


def main():
    p = argparse.ArgumentParser(description="REINFORCE on GridWorld.")
    p.add_argument("--updates", type=int, default=150,
                   help="number of policy-gradient updates")
    p.add_argument("--batch_steps", type=int, default=1500,
                   help="collect at least this many env steps per update")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--ent_coef", type=float, default=0.01,
                   help="entropy bonus: keeps the policy exploring")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default="reinforce.png")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "gpu"])
    args = p.parse_args()

    if args.device:
        babytorch.set_device(args.device)
    C.set_seed(args.seed)
    print(f"Device: {babytorch.device()}")

    env = GridWorld()
    policy = C.PolicyNet(env.obs_dim, env.n_actions, args.hidden)
    optimizer = Adam(policy.parameters(), learning_rate=args.lr)
    print(f"Policy: {policy.num_parameters():,} parameters\n")

    returns_history = []
    for update in range(args.updates):
        # --- collect a batch of whole episodes -----------------------------
        # A single episode is a noisy estimate of the gradient; averaging
        # over a batch of them (and normalising the returns *across* the
        # batch) is what makes plain policy gradient stable.  It also fixes
        # the trap of normalising inside one episode, which would invent a
        # fake "good vs bad" signal even for an episode that never won.
        b_states, b_actions, b_returns = [], [], []
        steps = 0
        while steps < args.batch_steps:
            states, actions, rewards = run_episode(env, policy)
            b_states.append(states)
            b_actions.append(actions)
            b_returns.append(C.discounted_returns(rewards, args.gamma))
            returns_history.append(float(np.sum(rewards)))
            steps += len(rewards)

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

        if update % 10 == 0 or update == args.updates - 1:
            recent = returns_history[-50:]
            print(f"update {update:4d} | episodes {len(returns_history):5d} | "
                  f"avg return {np.mean(recent):7.2f} | "
                  f"solved {100 * np.mean([r > 0 for r in recent]):5.1f}%")

    final = np.mean(returns_history[-50:])
    print(f"\nFinal 50-episode average return: {final:.2f}  "
          f"(a short win scores near +1..+4; a lost episode near -100)")
    C.plot_returns(returns_history, args.plot, "REINFORCE on GridWorld")


# Keep this handy for the tests: a single self-contained training run.
def train(updates=150, batch_steps=1500, gamma=0.99, lr=3e-3, ent_coef=0.01,
          hidden=128, seed=0, env=None):
    C.set_seed(seed)
    env = env or GridWorld()
    policy = C.PolicyNet(env.obs_dim, env.n_actions, hidden)
    optimizer = Adam(policy.parameters(), learning_rate=lr)
    returns_history = []
    for _ in range(updates):
        b_states, b_actions, b_returns = [], [], []
        steps = 0
        while steps < batch_steps:
            states, actions, rewards = run_episode(env, policy)
            b_states.append(states)
            b_actions.append(actions)
            b_returns.append(C.discounted_returns(rewards, gamma))
            returns_history.append(float(np.sum(rewards)))
            steps += len(rewards)
        advantages = C.to_tensor(C.normalize(np.concatenate(b_returns)))
        dist = C.Categorical(policy(C.to_tensor(np.concatenate(b_states))))
        loss = (-(dist.log_prob(np.concatenate(b_actions)) * advantages).mean()
                - ent_coef * dist.entropy().mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return policy, returns_history


if __name__ == "__main__":
    main()
