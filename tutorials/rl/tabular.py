"""Classical (tabular) reinforcement learning — RL before the networks.

Four foundational algorithms on GridWorld, with no neural network in
sight: just a *table* of values, updated by the Bellman equation. This is
the on-ramp chapter 9 explains, and it makes DQN (chapter 11) obvious --
DQN is what you do when the table is too big to store and you replace it
with a network that generalises.

Two families:

* **Dynamic programming** (``value_iteration``, ``policy_iteration``) --
  *model-based*: they are handed the rules (``env.transition``) and sweep
  every state to compute the optimal values exactly.
* **Temporal-difference control** (``sarsa``, ``q_learning``) --
  *model-free*: they don't know the rules, they just play
  (``env.reset``/``env.step``) and bootstrap each estimate off the next.

Run it::

    python tabular.py
"""

import numpy as np

from gridworld import GridWorld

ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}   # matches ACTIONS: up, down, left, right


# ---------------------------------------------------------------------------
# Model-based: dynamic programming (knows env.transition)
# ---------------------------------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-6):
    """Repeatedly apply the Bellman *optimality* backup until values settle.

        V(s) = max_a [ r(s,a) + gamma · V(s') ]

    Because GridWorld is deterministic, the expectation over next states is
    just a single ``(s', r)`` per action. Returns ``(V, greedy_policy)``.
    """
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


def policy_iteration(env, gamma=0.99, theta=1e-6):
    """Alternate two exact steps until the policy stops changing:

    1. **evaluation** -- solve V for the *current* policy (Bellman
       *expectation* backup, no max);
    2. **improvement** -- act greedily with respect to that V.

    Returns ``(V, policy)`` -- the same optimum value iteration finds,
    reached a different way.
    """
    policy = {s: 0 for s in env.states()}
    V = {s: 0.0 for s in env.states()}
    while True:
        while True:                                  # policy evaluation
            delta = 0.0
            for s in env.states():
                if env.is_terminal(s):
                    continue
                ns, r = env.transition(s, policy[s])
                v = r + gamma * V[ns]
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < theta:
                break
        stable = True                                # policy improvement
        for s in env.states():
            if env.is_terminal(s):
                continue
            best = _greedy_action(env, V, gamma, s)
            if best != policy[s]:
                stable = False
            policy[s] = best
        if stable:
            return V, policy


def _greedy_action(env, V, gamma, s):
    q = [r + gamma * V[ns] for ns, r in
         (env.transition(s, a) for a in range(env.n_actions))]
    return int(np.argmax(q))


def _greedy_from_v(env, V, gamma):
    return {s: (_greedy_action(env, V, gamma, s) if not env.is_terminal(s) else 0)
            for s in env.states()}


# ---------------------------------------------------------------------------
# Model-free: temporal-difference control (only plays the game)
# ---------------------------------------------------------------------------

def _epsilon_greedy(Q, s, n_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[s]))


def sarsa(env, episodes=500, gamma=0.99, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Update toward the action you *actually* take next:

        Q(s,a) += alpha · [ r + gamma · Q(s', a') − Q(s,a) ]

    ``a'`` is drawn from the same ε-greedy policy, so SARSA learns the value
    of the exploratory behaviour it follows. Returns ``(Q, greedy_policy)``.
    """
    Q = {s: np.zeros(env.n_actions) for s in env.states()}
    for _ in range(episodes):
        env.reset()
        s = env.pos                                  # the discrete cell IS the state
        a = _epsilon_greedy(Q, s, env.n_actions, epsilon)
        done = False
        while not done:
            _, r, done, _ = env.step(a)
            ns = env.pos
            na = _epsilon_greedy(Q, ns, env.n_actions, epsilon)
            target = r + (0.0 if done else gamma * Q[ns][na])
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = ns, na
    return Q, _greedy_from_q(env, Q)


def q_learning(env, episodes=500, gamma=0.99, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Update toward the *best* next action, whatever
    you actually do:

        Q(s,a) += alpha · [ r + gamma · maxₐ' Q(s', a') − Q(s,a) ]

    It learns the optimal (greedy) values while still exploring ε-greedily --
    the tabular ancestor of DQN. Returns ``(Q, greedy_policy)``.
    """
    Q = {s: np.zeros(env.n_actions) for s in env.states()}
    for _ in range(episodes):
        env.reset()
        s = env.pos
        done = False
        while not done:
            a = _epsilon_greedy(Q, s, env.n_actions, epsilon)
            _, r, done, _ = env.step(a)
            ns = env.pos
            target = r + (0.0 if done else gamma * np.max(Q[ns]))
            Q[s][a] += alpha * (target - Q[s][a])
            s = ns
    return Q, _greedy_from_q(env, Q)


def _greedy_from_q(env, Q):
    return {s: int(np.argmax(Q[s])) for s in env.states()}


# ---------------------------------------------------------------------------
# Pretty-printing a policy as arrows on the board
# ---------------------------------------------------------------------------

def render_policy(env, policy):
    for r in range(env.rows):
        line = []
        for c in range(env.cols):
            if (r, c) == env.goal:
                line.append("G")
            elif (r, c) in env.obstacles:
                line.append("#")
            else:
                line.append(ARROWS[policy[(r, c)]])
        print(" ".join(line))
    print()


if __name__ == "__main__":
    np.random.seed(0)
    env = GridWorld()

    print("Value iteration (model-based, exact):")
    V, pi_vi = value_iteration(env)
    render_policy(env, pi_vi)

    print("Policy iteration (model-based, exact) — same optimum:")
    _, pi_pi = policy_iteration(env)
    render_policy(env, pi_pi)

    print("SARSA (model-free, on-policy, 500 episodes):")
    _, pi_sarsa = sarsa(env)
    render_policy(env, pi_sarsa)

    print("Q-learning (model-free, off-policy, 500 episodes):")
    _, pi_q = q_learning(env)
    render_policy(env, pi_q)

    # On a maze this small, the model-free methods should recover the
    # exact optimal policy on the cells that matter (those on good paths).
    reachable = [s for s in env.states() if not env.is_terminal(s)]
    agree = np.mean([pi_q[s] == pi_vi[s] for s in reachable])
    print(f"Q-learning agrees with the exact optimum on "
          f"{100 * agree:.0f}% of states.")
