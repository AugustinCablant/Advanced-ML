import numpy as np
import gymnasium as gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt

class BernoulliBanditEnv(gym.Env):
    """
    Bernoulli Bandit environment.
    Each arm generates a reward with a Bernoulli distribution.
    """
    def __init__(self, means):
        self.p_dist = means
        self.n_bandits = len(means)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)  # No observations
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        reward = np.random.binomial(1, self.p_dist[action])  # Bernoulli reward
        done = True
        return [0], reward, done, {}

    def _reset(self):
        return [0]

    def _render(self, mode='human', close=False):
        pass

    def get_action_set(self):
      """
      Generates a set of vectors in dimension self.d
      """
      return list(np.arange(self.n_bandits))