"""Reinforcement-learning tests: the games, the machinery, and -- the
point of it all -- that each agent actually *learns*.

The learning tests train on a deliberately tiny open GridWorld with a
fixed seed, so they stay fast and deterministic while still exercising the
whole pipeline (rollout -> advantage/target -> gradient step).
"""

import os
import sys

import numpy as np
import pytest

import babytorch
from babytorch import Tensor
from babytorch.backend import xp

# The RL tutorial lives in tutorials/rl; put it on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tutorials", "rl"))
import common as C          # noqa: E402
import gridworld            # noqa: E402
import snake                # noqa: E402
import reinforce            # noqa: E402
import actor_critic         # noqa: E402
import dqn                  # noqa: E402
import ppo                  # noqa: E402


def tiny_env():
    """A 4x4 open grid: every agent should master it in seconds."""
    return gridworld.GridWorld(shape=(4, 4), obstacles=[], max_steps=25)


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------

def test_gridworld_reset_and_observation():
    env = gridworld.GridWorld()
    obs, info = env.reset()
    assert obs.shape == (env.obs_dim,) == (6,)
    assert obs.dtype == np.float32 and info == {}
    assert env.n_actions == 4


def test_gridworld_goal_gives_bonus_and_ends():
    env = gridworld.GridWorld(shape=(2, 2), obstacles=[], start=(0, 0),
                              goal=(1, 1), goal_reward=10.0)
    env.reset()
    _, r1, done1, _ = env.step(1)          # down: (0,0) -> (1,0)
    assert r1 == -1.0 and not done1
    obs, r2, done2, info = env.step(3)     # right: (1,0) -> (1,1) = goal
    assert r2 == 9.0 and done2 and info["reached_goal"]  # -1 step + 10 bonus


def test_gridworld_walls_block_but_cost():
    # An obstacle right of the start: moving into it wastes the step.
    env = gridworld.GridWorld(shape=(3, 3), obstacles=[(0, 1)], start=(0, 0))
    env.reset()
    obs, reward, done, _ = env.step(3)     # right, into the wall
    assert reward == -1.0 and not done
    assert env.pos == (0, 0)               # did not move
    assert obs[3] == 1.0                   # "blocked to the right" flag is set


def test_gridworld_truncates_at_max_steps():
    env = gridworld.GridWorld(shape=(5, 5), max_steps=3)
    env.reset()
    for _ in range(2):
        _, _, done, _ = env.step(0)        # bump the top wall, going nowhere
        assert not done
    _, _, done, info = env.step(0)
    assert done and info["truncated"] and not info["reached_goal"]


def test_snake_features_and_death():
    game = snake.SnakeGame(rows=6, cols=6, obs_mode="features", seed=0)
    obs, _ = game.reset()
    assert obs.shape == (11,) and game.n_actions == 3
    # Facing right from the middle; turning left twice then going is a U-turn
    # into itself -- but relative actions forbid a direct reversal, so drive
    # straight into the wall instead to check the death reward.
    for _ in range(10):
        obs, reward, done, info = game.step(0)   # keep going straight
        if done:
            break
    assert done and reward == -10.0


def test_snake_grid_observation_shape_and_values():
    game = snake.SnakeGame(rows=6, cols=6, obs_mode="grid", seed=0)
    obs, _ = game.reset()
    assert obs.shape == (1, 8, 8) == game.obs_shape
    assert obs.min() == -1.0 and obs.max() == 1.0   # walls/body -1, food +1
    assert (obs == 1.0).sum() == 1                  # exactly one food cell


def test_snake_eating_grows_and_scores():
    game = snake.SnakeGame(rows=5, cols=5, obs_mode="features", seed=0)
    game.reset()
    game.food = game.snake[0]            # put food under the head-to-be
    # Force the head onto the food by placing it one step ahead.
    dr, dc = snake.DIRECTIONS[game.dir_idx]
    game.food = (game.snake[0][0] + dr, game.snake[0][1] + dc)
    length_before = len(game.snake)
    _, reward, _, info = game.step(0)
    assert reward == snake.REWARD_FOOD and info["score"] == 1
    assert len(game.snake) == length_before + 1     # grew by one


# ---------------------------------------------------------------------------
# common.py machinery
# ---------------------------------------------------------------------------

def test_categorical_samples_scores_and_differentiates():
    C.set_seed(0)
    policy = C.PolicyNet(6, 4, hidden=16)
    obs = C.to_tensor(np.random.randn(8, 6))
    dist = C.Categorical(policy(obs))
    actions = dist.sample()
    assert actions.shape == (8,) and actions.min() >= 0 and actions.max() < 4
    logp, ent = dist.log_prob(actions), dist.entropy()
    assert logp.shape == (8,) and ent.shape == (8,)
    for p in policy.parameters():
        p.grad = None
    (-(logp).mean()).backward()
    assert all(p.grad is not None for p in policy.parameters())


def test_discounted_returns_and_gae():
    assert np.allclose(C.discounted_returns([-1, -1, -1], 1.0), [-3, -2, -1])
    assert np.allclose(C.discounted_returns([0, 0, 1], 0.5), [0.25, 0.5, 1.0])
    # With a perfect zero critic and lam=1, advantage == Monte-Carlo return.
    adv, ret = C.compute_gae([1.0, 1.0], [0.0, 0.0], gamma=1.0, lam=1.0)
    assert np.allclose(adv, [2, 1]) and np.allclose(ret, [2, 1])


def test_huber_loss_value_and_gradient():
    pred = Tensor(np.array([0.0, 5.0], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([0.5, 0.0], dtype=np.float32))
    loss = C.huber_loss(pred, target, delta=1.0)
    assert abs(loss.item() - 2.3125) < 1e-5     # (0.125 + 4.5) / 2
    loss.backward()
    # gradient is clamped to [-1, 1] in the linear tail (the +5 error)
    assert np.allclose(babytorch.to_numpy(pred.grad), [-0.25, 0.5])


def test_elementwise_min_max_clip_and_grad():
    a = Tensor(np.array([1.0, 5.0, 3.0], dtype=np.float32), requires_grad=True)
    b = Tensor(np.array([4.0, 2.0, 3.0], dtype=np.float32))
    assert np.allclose(babytorch.to_numpy(C.maximum(a, b).data), [4, 5, 3])
    assert np.allclose(babytorch.to_numpy(C.minimum(a, b).data), [1, 2, 3])
    assert np.allclose(babytorch.to_numpy(C.clip_tensor(a, 2.0, 4.0).data), [2, 4, 3])
    C.minimum(a, b).sum().backward()
    assert np.allclose(babytorch.to_numpy(a.grad), [1, 0, 1])   # grad -> the min side


def test_replay_buffer_and_target_updates():
    buf = C.ReplayBuffer(50)
    for i in range(30):
        buf.push(np.zeros(6, np.float32), i % 4, -1.0, np.ones(6, np.float32), 0.0)
    s, a, r, s2, d = buf.sample(8)
    assert s.shape == (8, 6) and a.shape == (8,) and d.shape == (8,)

    src, tgt = C.QNet(6, 4, 8), C.QNet(6, 4, 8)
    C.hard_update(tgt, src)
    for p, q in zip(tgt.parameters(), src.parameters()):
        assert np.allclose(babytorch.to_numpy(p.data), babytorch.to_numpy(q.data))


# ---------------------------------------------------------------------------
# The point of it all: each agent learns
# ---------------------------------------------------------------------------

def test_reinforce_learns_gridworld():
    policy, _ = reinforce.train(updates=40, batch_steps=500, seed=0, env=tiny_env())
    _, solved = C.evaluate(tiny_env(), policy, episodes=20)
    assert solved >= 0.8


def test_actor_critic_learns_gridworld():
    actor, _, _ = actor_critic.train(updates=40, batch_steps=500, seed=0, env=tiny_env())
    _, solved = C.evaluate(tiny_env(), actor, episodes=20)
    assert solved >= 0.8


def test_dqn_learns_gridworld():
    q_net, _ = dqn.train(episodes=150, seed=0, env=tiny_env())
    _, solved = C.evaluate(tiny_env(), q_net, episodes=20)
    assert solved >= 0.8


def test_ppo_learns_gridworld():
    actor, _, _ = ppo.train(updates=25, batch_steps=500, minibatch=128, seed=0,
                            env=tiny_env())
    _, solved = C.evaluate(tiny_env(), actor, episodes=20)
    assert solved >= 0.8


def test_dqn_learns_snake_features():
    env = snake.SnakeGame(rows=6, cols=6, obs_mode="features")
    q_net, history = dqn.train(episodes=200, env=env, eps_decay=2000, seed=0)
    # Clearly better late than early: surviving longer and eating more.
    assert np.mean(history[-30:]) > np.mean(history[:30]) + 3.0


def test_dqn_grid_convnet_runs():
    # The ConvNet path is slow to *master* Snake, but must at least run end
    # to end: build a conv Q-net, fill the buffer, take real gradient steps.
    env = snake.SnakeGame(rows=5, cols=5, obs_mode="grid")
    net_fn = lambda e: C.ConvNet(e.obs_shape, e.n_actions, hidden=32)
    q_net, history = dqn.train(episodes=6, env=env, net_fn=net_fn,
                               batch_size=16, warmup=16, seed=0)
    assert len(history) == 6
