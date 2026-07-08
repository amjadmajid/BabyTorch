"""Actor-Critic: REINFORCE with a critic that learns the baseline.

REINFORCE weights each action by its return minus a *constant* baseline
(the batch mean). That works, but a good action in a bad state and a bad
action in a good state can look the same. The fix is a **critic**: a
second network ``V(s)`` that learns how much reward to *expect* from a
state. Now each action is judged by its **advantage** --

    advantage = return - V(state)

-- "how much better than expected did this turn out?". Actions with
positive advantage are made more likely; the baseline is now tuned to
each state, so the policy gradient is much less noisy.

Two networks, two losses, one shared rollout:

* the **actor** (policy) minimises ``-(log_prob * advantage)``;
* the **critic** (value) minimises ``(return - V(state))^2``, learning to
  predict the returns it is being asked to explain.

Ported from the A2C agent in
https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch

    python actor_critic.py
    python actor_critic.py --updates 120 --plot ac.png
    BABYTORCH_DEVICE=cpu python actor_critic.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import babytorch
from babytorch.backend import to_numpy
from babytorch.optim import Adam

import common as C
from gridworld import GridWorld


def run_episode(env, policy):
    """Play one episode by sampling from the policy (no gradients yet)."""
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    done = False
    with babytorch.no_grad():
        while not done:
            logits = policy(C.to_tensor(obs).reshape(1, -1))
            action = int(C.Categorical(logits).sample()[0])
            next_obs, reward, done, info = env.step(action)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
    return np.array(states, dtype=np.float32), np.array(actions), rewards


def train(updates=120, batch_steps=1500, gamma=0.99, actor_lr=3e-3,
          critic_lr=5e-3, ent_coef=0.01, hidden=128, seed=0, env=None,
          on_update=None):
    C.set_seed(seed)
    env = env or GridWorld()
    actor = C.PolicyNet(env.obs_dim, env.n_actions, hidden)
    critic = C.ValueNet(env.obs_dim, hidden)
    opt_actor = Adam(actor.parameters(), learning_rate=actor_lr)
    opt_critic = Adam(critic.parameters(), learning_rate=critic_lr)

    returns_history = []
    for update in range(updates):
        # --- collect a batch of whole episodes -----------------------------
        b_states, b_actions, b_returns = [], [], []
        steps = 0
        while steps < batch_steps:
            states, actions, rewards = run_episode(env, actor)
            b_states.append(states)
            b_actions.append(actions)
            b_returns.append(C.discounted_returns(rewards, gamma))
            returns_history.append(float(np.sum(rewards)))
            steps += len(rewards)
        states = C.to_tensor(np.concatenate(b_states))
        actions = np.concatenate(b_actions)
        returns = np.concatenate(b_returns)

        # --- the critic's prediction, and the advantage it implies ---------
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

        if on_update:
            on_update(update, returns_history)
    return actor, critic, returns_history


def main():
    p = argparse.ArgumentParser(description="Actor-Critic (A2C) on GridWorld.")
    p.add_argument("--updates", type=int, default=120)
    p.add_argument("--batch_steps", type=int, default=1500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--actor_lr", type=float, default=3e-3)
    p.add_argument("--critic_lr", type=float, default=5e-3)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default="actor_critic.png")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "gpu"])
    args = p.parse_args()

    if args.device:
        babytorch.set_device(args.device)
    print(f"Device: {babytorch.device()}\n")

    def report(update, history):
        if update % 10 == 0 or update == args.updates - 1:
            recent = history[-50:]
            print(f"update {update:4d} | episodes {len(history):5d} | "
                  f"avg return {np.mean(recent):7.2f} | "
                  f"solved {100 * np.mean([r > 0 for r in recent]):5.1f}%")

    actor, critic, history = train(
        updates=args.updates, batch_steps=args.batch_steps, gamma=args.gamma,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, ent_coef=args.ent_coef,
        hidden=args.hidden, seed=args.seed, on_update=report)

    print(f"\nFinal 50-episode average return: {np.mean(history[-50:]):.2f}")
    C.plot_returns(history, args.plot, "Actor-Critic on GridWorld")


if __name__ == "__main__":
    main()
