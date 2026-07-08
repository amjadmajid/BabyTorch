"""GridWorld: the simplest game an agent can learn to win.

A single agent lives on a small grid. Every step it moves up, down, left
or right; walls and obstacles block it (a blocked move wastes the step
but does not end the game). Reaching the goal ends the episode. Every
step costs ``-1`` and reaching the goal pays a ``+10`` bonus, so a short
successful path scores near zero, while an episode that never finds the
goal bottoms out around ``-max_steps``. Maximising return therefore
means one thing: **reach the goal by the shortest path**. The goal bonus
is what makes success stand out sharply from failure -- without it every
lost episode scores the same and there is nothing for a policy gradient
to grab onto.

This is a port of the GridWorld from
https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch
adapted to BabyTorch's conventions: pure NumPy (no Pygame), a Gymnasium-
style ``reset``/``step`` API, a normalised observation vector that a small
network can read, and -- the one real addition -- a ``max_steps`` cap so a
half-trained policy can never loop forever (episodic algorithms like
REINFORCE need every episode to *end*).

    env = GridWorld()
    obs, _ = env.reset()
    obs, reward, done, info = env.step(action_index)   # action in 0..3
    env.render()                                       # ASCII picture

The observation the agent sees at each position is six numbers::

    [blocked_up, blocked_down, blocked_left, blocked_right, row/H, col/W]

-- four "is there a wall that way?" flags plus where it is on the board.
That is enough to solve the maze and keeps the network tiny.
"""

import numpy as np

# Actions, in a fixed order.  Index 0..3 is what the agent chooses; the
# tuple is the (drow, dcol) it means.
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]

# Two mazes to choose from.  "wall" is the gentle default -- one vertical
# wall with a gap at the top and bottom, easy enough that a random policy
# stumbles onto the goal often enough to start learning.  "classic" is the
# harder maze from the original repo (a long forced detour); good for
# seeing how much a value baseline helps.  Both go from the top-left corner
# to the bottom-right.
MAZES = {
    "wall": [(1, 1), (2, 1), (3, 1)],
    "classic": [(0, 1), (1, 1), (2, 1), (3, 1), (2, 3), (3, 3), (4, 3)],
}
DEFAULT_OBSTACLES = MAZES["wall"]


class GridWorld:
    """A small grid maze with a start, a goal, and some walls."""

    def __init__(self, shape=(5, 5), obstacles=DEFAULT_OBSTACLES,
                 start=(0, 0), goal=None, max_steps=100, goal_reward=10.0):
        self.rows, self.cols = shape
        self.obstacles = set(tuple(o) for o in obstacles)
        self.start = tuple(start)
        # Default goal is the bottom-right corner.
        self.goal = tuple(goal) if goal is not None else (self.rows - 1, self.cols - 1)
        self.max_steps = max_steps
        self.goal_reward = goal_reward

        # Sizes the agent code reads to build its network.
        self.n_actions = len(ACTIONS)
        self.obs_dim = 6

        self.pos = self.start
        self.steps = 0

    # ------------------------------------------------------------------
    # The Gymnasium-style interface: reset() and step()
    # ------------------------------------------------------------------

    def reset(self):
        """Put the agent back at the start; return the first observation."""
        self.pos = self.start
        self.steps = 0
        return self._observe(), {}

    def step(self, action):
        """Apply one action. Returns ``(obs, reward, done, info)``.

        ``done`` is True when the agent reaches the goal *or* runs out of
        steps -- either way the episode is over and the caller should
        start a new one.
        """
        drow, dcol = ACTIONS[int(action)]
        target = (self.pos[0] + drow, self.pos[1] + dcol)

        # A move into a wall or off the board simply doesn't happen -- the
        # agent stays put (and still pays the -1 step cost below).
        if self._is_free(target):
            self.pos = target

        self.steps += 1
        reached = self.pos == self.goal
        truncated = self.steps >= self.max_steps
        done = reached or truncated

        # -1 every step, plus a +goal_reward bonus for reaching the goal.
        # Shorter successful path -> higher (closer to zero) return; a lost
        # episode just accumulates step costs down to about -max_steps.
        reward = -1.0 + (self.goal_reward if reached else 0.0)
        info = {"reached_goal": reached, "truncated": truncated and not reached}
        return self._observe(), reward, done, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_board(self, pos):
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols

    def _is_free(self, pos):
        """Can the agent stand here? (on the board and not an obstacle)."""
        return self._on_board(pos) and pos not in self.obstacles

    # ------------------------------------------------------------------
    # The model, laid bare -- for the tabular / dynamic-programming
    # methods of chapter 9 (value iteration, policy iteration).  Those
    # need to know the rules in advance: every state, and where each
    # action leads.  GridWorld is deterministic, so a transition is a
    # single (next_state, reward), not a distribution.
    # ------------------------------------------------------------------

    def states(self):
        """Every cell the agent can occupy (the whole state space)."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.obstacles]

    def is_terminal(self, state):
        """The goal is absorbing: the episode ends when you step onto it."""
        return tuple(state) == self.goal

    def transition(self, state, action):
        """Where does ``action`` lead from ``state``? -> (next_state, reward).

        Mirrors :meth:`step` exactly, but as a pure function of ``state``
        with no side effects -- a blocked move stays put, and the reward is
        ``-1`` plus the goal bonus when the move lands on the goal.
        """
        state = tuple(state)
        if self.is_terminal(state):
            return state, 0.0                    # nothing follows the goal
        drow, dcol = ACTIONS[int(action)]
        target = (state[0] + drow, state[1] + dcol)
        nxt = target if self._is_free(target) else state
        reward = -1.0 + (self.goal_reward if nxt == self.goal else 0.0)
        return nxt, reward

    def _observe(self):
        """The six-number view the agent gets (see the module docstring)."""
        blocked = [0.0 if self._is_free((self.pos[0] + dr, self.pos[1] + dc))
                   else 1.0 for dr, dc in ACTIONS]
        row = self.pos[0] / max(1, self.rows - 1)
        col = self.pos[1] / max(1, self.cols - 1)
        return np.array(blocked + [row, col], dtype=np.float32)

    def render(self):
        """Print the board: ``A`` = agent, ``G`` = goal, ``#`` = wall."""
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) == self.pos:
                    row.append("A")
                elif (r, c) == self.goal:
                    row.append("G")
                elif (r, c) in self.obstacles:
                    row.append("#")
                else:
                    row.append(".")
            print(" ".join(row))
        print()


if __name__ == "__main__":
    # A quick manual sanity check: walk the agent down and right.
    env = GridWorld()
    obs, _ = env.reset()
    env.render()
    for a in [1, 1, 1, 1, 3, 3, 3, 3]:      # down x4, right x4
        obs, reward, done, info = env.step(a)
        print(f"action={ACTION_NAMES[a]:5s} reward={reward:+.0f} done={done} "
              f"obs={obs.round(2)}")
        env.render()
        if done:
            print("reached goal!" if info["reached_goal"] else "out of steps")
            break
