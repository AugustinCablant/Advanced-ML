import numpy as np
from numpy.linalg import pinv

class EpsilonGreedy:
    """
    Implementation of the Epsilon-Greedy algorithm for multi-armed bandits.
    This algorithm balances exploration and exploitation by selecting random actions
    with a fixed or dynamically adjusted probability.

    Attributes:
        K (int): Number of actions (arms).
        FixedEpsilon (bool): If True, dynamically adjusts the exploration probability.
        eps (float): Initial probability of exploration (default is 0.1).
        m (int): Number of times to play each arm initially for exploration (default is 1).

    Methods:
        reset():
            Resets the internal state of the algorithm, including the reward estimates
            and counts for each arm.

        get_action(t):
            Selects an action based on the epsilon-greedy strategy, where an arm is either
            chosen randomly (exploration) or greedily based on the reward estimates (exploitation).

        receive_reward(chosen_arm, reward):
            Updates the reward estimate for the chosen arm based on the observed reward.

        name():
            Returns the name of the algorithm, including details of the exploration strategy.
    """

    def __init__(self, K, eps = 0.1, FixedEpsilon = False, m = 1):
        """
        Initializes the EpsilonGreedy algorithm.

        Args:
            K (int): Number of actions (arms).
            eps (float): Initial probability of exploration (default is 0.1).
            FixedEpsilon (bool): If True, dynamically adjusts the exploration probability.
            m (int): Number of times to play each arm initially for exploration (default is 1).
        """
        self.K = K
        self.eps = eps
        self.FixedEpsilon = FixedEpsilon
        self.m = m
        self.reset()

    def reset(self):
        """
        Resets the internal state of the algorithm.
        Initializes reward estimates and counts for each arm.
        """
        self.t = 0
        self.counts = np.zeros(self.K)  # Number of times each arm is selected
        self.q_values = np.zeros(self.K)  # Estimated rewards for each arm

    def get_numberPlayed(self):
        """ Return number of times this agent has been played. """
        return self.t
    
    def get_action(self, arms, t):
        """
        Selects an action based on the epsilon-greedy strategy.

        Args:
            t (int): The current time step.

        Returns:
            int: The index of the selected action (arm).
        """
        if t < self.m * self.K:  # Initial exploration phase
            return t % self.K
        else:
            if self.FixedEpsilon:
                self.eps = min(1, 5 * self.K / (t * (np.min(self.q_values) ** 2 + 1e-10)))
            if np.random.random() < self.eps:  # Exploration step
                index = np.random.randint(self.K)
                return arms[index]
            else:  # Exploitation step
                index = np.argmax(self.q_values)
                return arms[index]

    def receive_reward(self, chosen_arm, reward):
        """
        Updates the reward estimate for the chosen arm.

        Args:
            chosen_arm (int): The index of the chosen action (arm).
            reward (float): The observed reward for the chosen action.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.q_values[chosen_arm] += (reward - self.q_values[chosen_arm]) / n
        self.t += 1  # Increment time step

    def name(self):
        """
        Returns the name of the algorithm, including configuration details.

        Returns:
            str: The name of the algorithm, with details of epsilon and exploration strategy.
        """
        strategy = "Dynamic" if self.FixedEpsilon else "Fixed"
        return f"EpsilonGreedy({strategy} epsilon = {self.eps}, m = {self.m})"
