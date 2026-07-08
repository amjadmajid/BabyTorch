"""Snake: the same DQN, a harder game -- and eyes, if you want them.

GridWorld is a warm-up. Snake is a real arcade game: a growing body, food
to chase, and death by wall or by biting yourself. The agent is the exact
same DQN from ``dqn.py``; only the game changes -- which is the whole
point. It can see the board in two ways, and you pick with ``obs_mode``:

* ``"features"`` -- an 11-number hand-built summary: is there danger
  straight / right / left, which way am I heading, where is the food.
  A tiny MLP learns from this fast.
* ``"grid"``     -- the raw board as a small image ``(1, H, W)``: walls and
  body are ``-1``, the head ``-0.5``, the food ``+1``. Now the agent has to
  *see*, and the Q-network grows a couple of convolutions -- the same DQN
  with eyes instead of hand-fed features.

Actions are **relative** to the current heading -- go straight, turn
right, turn left -- so the snake can never reverse into itself.

Ported (and de-Pygame-d) from the two Snake agents in
https://github.com/amjadmajid/deep-reinforcement-learning-games-from-scratch

    python snake.py            # a random game, rendered as ASCII
"""

import numpy as np

# Headings in clockwise order, as (drow, dcol).  A "turn right" is +1 around
# this ring, a "turn left" is -1, so the snake turns relative to where it is
# already pointing -- and can never make an instant 180.
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]      # up, right, down, left

REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = 0.0

N_ACTIONS = 3          # 0 = straight, 1 = turn right, 2 = turn left


class SnakeGame:
    """A grid game of Snake with a Gymnasium-style ``reset``/``step`` API."""

    def __init__(self, rows=8, cols=8, obs_mode="features",
                 max_idle_steps=100, shaping=0.0, seed=None):
        assert obs_mode in ("features", "grid")
        self.rows, self.cols = rows, cols
        self.obs_mode = obs_mode
        self.max_idle_steps = max_idle_steps      # give up if food isn't eaten
        # Optional reward shaping: a small nudge for moving toward the food.
        # Off by default (faithful to the original game); turning it on makes
        # the sparse +10/-10 signal far easier to learn from, which the pixel
        # (grid) agent in particular needs.
        self.shaping = shaping
        if seed is not None:                      # else follow the global seed
            np.random.seed(seed)

        self.n_actions = N_ACTIONS
        self.obs_dim = 11                          # for the "features" MLP
        self.obs_shape = (1, rows + 2, cols + 2)   # for the "grid" ConvNet
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        r, c = self.rows // 2, self.cols // 2
        self.dir_idx = 1                           # heading right
        # Body laid out behind the head so the first move is legal.
        self.snake = [(r, c), (r, c - 1), (r, c - 2)]
        self.score = 0
        self.idle = 0
        self._place_food()
        return self._observe(), {}

    def step(self, action):
        # Turn relative to the current heading, then step one cell.
        if action == 1:
            self.dir_idx = (self.dir_idx + 1) % 4          # right
        elif action == 2:
            self.dir_idx = (self.dir_idx - 1) % 4          # left
        dr, dc = DIRECTIONS[self.dir_idx]
        old_head = self.snake[0]
        head = (old_head[0] + dr, old_head[1] + dc)

        self.idle += 1
        if self._hits_wall(head) or head in self.snake:
            return self._observe(), REWARD_DEATH, True, {"score": self.score}

        self.snake.insert(0, head)
        if head == self.food:                              # ate: grow + respawn
            self.score += 1
            self.idle = 0
            reward = REWARD_FOOD
            self._place_food()
        else:                                              # moved: drop the tail
            self.snake.pop()
            reward = REWARD_STEP
            # Shaping: reward getting closer to the food (Manhattan distance).
            if self.shaping:
                closer = (self._food_dist(old_head) - self._food_dist(head))
                reward += self.shaping * closer

        # Truncate a snake that is just going in circles without eating.
        truncated = self.idle >= self.max_idle_steps
        return self._observe(), reward, truncated, {"score": self.score}

    # ------------------------------------------------------------------
    def _hits_wall(self, cell):
        return not (0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols)

    def _food_dist(self, cell):
        return abs(cell[0] - self.food[0]) + abs(cell[1] - self.food[1])

    def _place_food(self):
        free = [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.snake]
        self.food = free[np.random.randint(len(free))] if free else self.snake[0]

    def _danger(self, turn):
        """Would turning by ``turn`` (0/+1/-1) and stepping kill us?"""
        d = DIRECTIONS[(self.dir_idx + turn) % 4]
        cell = (self.snake[0][0] + d[0], self.snake[0][1] + d[1])
        return 1.0 if (self._hits_wall(cell) or cell in self.snake) else 0.0

    def _observe(self):
        if self.obs_mode == "grid":
            return self._grid_obs()
        head, food = self.snake[0], self.food
        one_hot = [0.0, 0.0, 0.0, 0.0]
        one_hot[self.dir_idx] = 1.0
        return np.array([
            self._danger(0), self._danger(+1), self._danger(-1),      # dangers
            *one_hot,                                                  # heading
            float(food[0] < head[0]), float(food[0] > head[0]),       # food row
            float(food[1] < head[1]), float(food[1] > head[1]),       # food col
        ], dtype=np.float32)

    def _grid_obs(self):
        """The board as a small image with a wall border: (1, rows+2, cols+2)."""
        g = np.zeros((self.rows + 2, self.cols + 2), dtype=np.float32)
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = -1.0     # walls
        for (r, c) in self.snake:
            g[r + 1, c + 1] = -1.0                         # body
        hr, hc = self.snake[0]
        g[hr + 1, hc + 1] = -0.5                           # head, marked apart
        g[self.food[0] + 1, self.food[1] + 1] = 1.0        # food
        return g[None, :, :]                               # add channel axis

    def render(self):
        chars = {}
        for (r, c) in self.snake:
            chars[(r, c)] = "o"
        chars[self.snake[0]] = "H"
        chars[self.food] = "*"
        print("+" + "-" * self.cols + "+")
        for r in range(self.rows):
            print("|" + "".join(chars.get((r, c), " ") for c in range(self.cols)) + "|")
        print("+" + "-" * self.cols + f"+  score={self.score}")


if __name__ == "__main__":
    game = SnakeGame(rows=8, cols=8, seed=0)
    obs, _ = game.reset()
    game.render()
    print("features:", obs)
    done = False
    while not done:
        obs, reward, done, info = game.step(np.random.randint(3))
        if reward != 0:
            game.render()
            print(f"reward={reward:+.0f} done={done} score={info['score']}")
