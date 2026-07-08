"""Play Snake with DQN -- with hand-built features, or with convolutions.

This is not a new algorithm: it is ``dqn.py`` pointed at ``snake.py``.
The lesson is that the agent doesn't change when the game does -- only the
network's *eyes* do:

    python snake_dqn.py --net features    # 11-number summary -> tiny MLP
    python snake_dqn.py --net grid        # raw board -> ConvNet (slower)
    BABYTORCH_DEVICE=cpu python snake_dqn.py --net features

``--net features`` learns quickly. ``--net grid`` feeds the raw board to a
small convolutional Q-network -- the more honest, harder task; in pure
NumPy/CuPy it trains slowly, so keep the grid small and be patient.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import babytorch

import common as C
import dqn
from snake import SnakeGame


def evaluate_snake(make_env, net, episodes=30):
    """Greedy play: average score (food eaten) and return over some games."""
    scores, returns = [], []
    for _ in range(episodes):
        env = make_env()
        obs, _ = env.reset()
        done, total, info = False, 0.0, {}
        with babytorch.no_grad():
            while not done:
                q = net(C.to_tensor(obs[None]))
                obs, reward, done, info = env.step(int(q.argmax(axis=1)[0]))
                total += reward
        scores.append(info.get("score", 0))
        returns.append(total)
    return float(np.mean(scores)), float(np.mean(returns))


def main():
    p = argparse.ArgumentParser(description="DQN plays Snake.")
    p.add_argument("--net", choices=["features", "grid"], default="features",
                   help="'features' = MLP over 11 features; 'grid' = ConvNet")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps_decay", type=float, default=4000,
                   help="exploration decay in steps (smaller = explore less)")
    p.add_argument("--shaping", type=float, default=0.0,
                   help="reward for moving toward food (helps the grid net)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default="snake_dqn.png")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "gpu"])
    args = p.parse_args()

    if args.device:
        babytorch.set_device(args.device)
    print(f"Device: {babytorch.device()}  |  net: {args.net}\n")

    mode = "grid" if args.net == "grid" else "features"
    make_env = lambda: SnakeGame(args.rows, args.cols, obs_mode=mode,
                                 shaping=args.shaping)
    env = make_env()

    # The only thing the network type changes: an MLP for features, a
    # ConvNet for the raw grid.  The DQN training loop is identical.
    net_fn = None
    if args.net == "grid":
        net_fn = lambda e: C.ConvNet(e.obs_shape, e.n_actions, args.hidden)

    def report(ep, history, eps):
        if ep % 25 == 0 or ep == args.episodes - 1:
            recent = history[-25:]
            print(f"episode {ep:4d} | avg return {np.mean(recent):7.2f} | "
                  f"epsilon {eps:.2f}")

    q_net, history = dqn.train(
        episodes=args.episodes, env=env, net_fn=net_fn, hidden=args.hidden,
        lr=args.lr, eps_decay=args.eps_decay, seed=args.seed, on_episode=report)

    avg_score, avg_return = evaluate_snake(make_env, q_net, episodes=30)
    print(f"\nGreedy evaluation over 30 games: "
          f"average score {avg_score:.2f} food, average return {avg_return:.1f}")
    C.plot_returns(history, args.plot, f"DQN on Snake ({args.net})", window=25)


if __name__ == "__main__":
    main()
