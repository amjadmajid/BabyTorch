"""PPO: the policy-gradient method everyone actually uses.

Actor-Critic works, but it is wasteful and twitchy: each batch of
experience is thrown away after a single gradient step, and nothing stops
that step from being so large it wrecks the policy. **Proximal Policy
Optimization** (Schulman et al., 2017) fixes both, and is the workhorse
behind most modern RL -- including the RLHF that fine-tunes language
models.

Two ideas on top of Actor-Critic:

* **The clipped objective.** Track the *ratio* between the new policy and
  the one that collected the data, ``r = pi_new(a|s) / pi_old(a|s)``.
  Multiply it by the advantage, but **clip** it to ``[1-eps, 1+eps]`` so a
  single update can never move the policy too far. That safety rail lets
  us...
* **...reuse each batch for several epochs** of minibatch updates, instead
  of one step and throw it away -- far more sample-efficient.

Advantages come from :func:`common.compute_gae` (GAE), a smoother estimate
than raw returns. This algorithm is not in the original repo -- it is the
modern capstone on top of its REINFORCE -> A2C progression.

    python ppo.py
    python ppo.py --updates 60 --plot ppo.png
    BABYTORCH_DEVICE=cpu python ppo.py
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


def collect_batch(env, actor, critic, batch_steps, gamma, lam):
    """Run episodes until ``batch_steps`` steps, recording everything PPO
    needs: states, actions, the log-probs *the current policy* gave them
    (``old`` log-probs, frozen for the update), GAE advantages and value
    targets."""
    states, actions, old_logps, advantages, returns, ep_returns = [], [], [], [], [], []
    steps = 0
    while steps < batch_steps:
        obs, _ = env.reset()
        s_ep, a_ep, r_ep, v_ep, lp_ep = [], [], [], [], []
        done, info = False, {}
        with babytorch.no_grad():
            while not done:
                st = C.to_tensor(obs).reshape(1, -1)
                dist = C.Categorical(actor(st))
                action = int(dist.sample()[0])
                s_ep.append(obs)
                a_ep.append(action)
                r_ep.append(0.0)                 # reward filled in after the step
                v_ep.append(float(critic(st).item()))
                lp_ep.append(float(dist.log_prob([action]).item()))
                obs, reward, done, info = env.step(action)
                r_ep[-1] = reward
            # If the episode was cut off (not a real terminal), bootstrap the
            # tail with the critic's value of where we stopped.
            last_v = (float(critic(C.to_tensor(obs).reshape(1, -1)).item())
                      if info.get("truncated") else 0.0)

        adv, ret = C.compute_gae(r_ep, v_ep, gamma, lam, last_v)
        states.append(np.array(s_ep, dtype=np.float32))
        actions.append(np.array(a_ep))
        old_logps.append(np.array(lp_ep, dtype=np.float32))
        advantages.append(adv)
        returns.append(ret)
        ep_returns.append(float(np.sum(r_ep)))
        steps += len(r_ep)

    return (np.concatenate(states), np.concatenate(actions),
            np.concatenate(old_logps), np.concatenate(advantages),
            np.concatenate(returns), ep_returns)


def ppo_update(actor, critic, optimizer, batch, clip_eps=0.2, epochs=4,
               minibatch=256, ent_coef=0.01, vf_coef=0.5, grad_clip=1.0):
    """Several epochs of clipped-surrogate minibatch updates on one batch."""
    states, actions, old_logps, advantages, returns, _ = batch
    advantages = C.normalize(advantages)                 # steadier gradients
    params = list(actor.parameters()) + list(critic.parameters())
    n = len(states)

    for _ in range(epochs):
        order = np.random.permutation(n)
        for start in range(0, n, minibatch):
            idx = order[start:start + minibatch]
            s = C.to_tensor(states[idx])

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

            value_loss = ((C.to_tensor(returns[idx]) - critic(s)) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            C.clip_grad_value(params, grad_clip)
            optimizer.step()


def train(updates=80, batch_steps=1500, gamma=0.99, lam=0.95, lr=3e-3,
          clip_eps=0.2, epochs=4, minibatch=256, ent_coef=0.01, vf_coef=0.5,
          hidden=128, seed=0, env=None, on_update=None):
    C.set_seed(seed)
    env = env or GridWorld()
    actor = C.PolicyNet(env.obs_dim, env.n_actions, hidden)
    critic = C.ValueNet(env.obs_dim, hidden)
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()),
                     learning_rate=lr)

    history = []
    for update in range(updates):
        batch = collect_batch(env, actor, critic, batch_steps, gamma, lam)
        history.extend(batch[5])
        ppo_update(actor, critic, optimizer, batch, clip_eps, epochs,
                   minibatch, ent_coef, vf_coef)
        if on_update:
            on_update(update, history)
    return actor, critic, history


def main():
    p = argparse.ArgumentParser(description="PPO on GridWorld.")
    p.add_argument("--updates", type=int, default=80)
    p.add_argument("--batch_steps", type=int, default=1500)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=256)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default="ppo.png")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "gpu"])
    args = p.parse_args()

    if args.device:
        babytorch.set_device(args.device)
    print(f"Device: {babytorch.device()}\n")

    def report(update, history):
        if update % 5 == 0 or update == args.updates - 1:
            recent = history[-50:]
            print(f"update {update:4d} | episodes {len(history):5d} | "
                  f"avg return {np.mean(recent):7.2f} | "
                  f"solved {100 * np.mean([r > 0 for r in recent]):5.1f}%")

    actor, critic, history = train(
        updates=args.updates, batch_steps=args.batch_steps, gamma=args.gamma,
        lam=args.lam, lr=args.lr, clip_eps=args.clip_eps, epochs=args.epochs,
        minibatch=args.minibatch, ent_coef=args.ent_coef, hidden=args.hidden,
        seed=args.seed, on_update=report)

    avg_return, solved = C.evaluate(GridWorld(), actor, episodes=50)
    print(f"\nGreedy evaluation over 50 episodes: "
          f"avg return {avg_return:.2f}, solved {100 * solved:.0f}%")
    C.plot_returns(history, args.plot, "PPO on GridWorld")


if __name__ == "__main__":
    main()
