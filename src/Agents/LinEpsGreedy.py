from src.utils.imports import *

class LinEpsilonGreedy:
    """
    Implementation of the Linear Epsilon-Greedy algorithm for contextual bandits.
    This algorithm balances exploration and exploitation by selecting random actions
    with a fixed or dynamic probability and leveraging linear regression for parameter estimation.

    Attributes:
        d (int): Dimensionality of the feature space.
        lambda_reg (float): Regularization parameter for ridge regression.
        m (int): Number of times to play each arm initially for exploration (default is 1).
        eps (float): Initial probability of exploration (default is 0.1).
        SGD_Up (bool): If True, uses Stochastic Gradient Descent for parameter updates (default is False).
        FixedEpsilon (bool): If True, dynamically adjusts the exploration probability (default is False).
        other_option (optional): Placeholder for additional options (default is None).

    Methods:
        reset():
            Resets the internal state of the algorithm, including the parameter estimates,
            covariance matrices, and exploration counts.

        get_action(arms, t):
            Selects an action based on the epsilon-greedy strategy, where an arm is either 
            chosen randomly (exploration) or greedily based on the parameter estimates (exploitation).

        receive_reward(chosen_arm, reward, eta=0.01):
            Updates the model parameters based on the observed reward. 
            Can use either SGD or least squares updates depending on the configuration.

        name():
            Returns the name of the algorithm, including details of the exploration strategy.
    """

    def __init__(self, d, lambda_reg, m=1, eps=0.1, SGD_Up=False, FixedEpsilon=False, other_option=None):
        """
        Initializes the LinEpsilonGreedy algorithm.

        Args:
            d (int): Dimensionality of the feature space.
            lambda_reg (float): Regularization parameter for ridge regression.
            m (int): Number of times to play each arm initially for exploration (default is 1).
            eps (float): Initial probability of exploration (default is 0.1).
            SGD_Up (bool): If True, uses Stochastic Gradient Descent for parameter updates (default is False).
            FixedEpsilon (bool): If True, dynamically adjusts the exploration probability (default is False).
            other_option (optional): Placeholder for additional options (default is None).
        """
        self.eps = eps
        self.d = d
        self.lambda_reg = lambda_reg
        self.SGD_Up = SGD_Up
        self.m = m
        self.FixedEpsilon = FixedEpsilon
        self.reset()

    def reset(self):
        """
        Resets the internal state of the algorithm.
        Reinitializes parameter estimates, covariance matrices, and exploration counts.
        """
        self.t = 0
        self.hat_theta = np.ones(self.d)  # Estimated parameters
        self.cov = self.lambda_reg * np.identity(self.d)  # Covariance matrix
        self.invcov = np.identity(self.d)  # Inverse covariance matrix
        self.b_t = np.zeros(self.d)  # Accumulated reward-weighted features

    def get_action(self, arms, t):
        """
        Selects an action based on the epsilon-greedy strategy.

        Args:
            arms (numpy.ndarray): A 2D array where each row represents an action's feature vector.
            t (int): The current time step.

        Returns:
            numpy.ndarray: The feature vector of the selected action.
        """
        K, _ = arms.shape
        if t < self.m * K:  # Initial exploration phase
            index = t % K
            return arms[index]
        else:
            expl = np.random.random()  # Random value for exploration check
            if self.FixedEpsilon:
                self.eps = np.min(1, K / (self.t * (self.d ** 2)))
            if expl < self.eps:  # Exploration step
                action = arms[np.random.randint(K)]
            else:  # Exploitation step
                index = np.argmax([np.dot(arm, self.hat_theta) for arm in arms])
                action = arms[index]
            return action

    def receive_reward(self, chosen_arm, reward, eta=0.01):
        """
        Updates the internal parameters based on the observed reward.

        Args:
            chosen_arm (numpy.ndarray): The feature vector of the chosen action.
            reward (float): The observed reward for the chosen action.
            eta (float): Learning rate for SGD updates (default is 0.01).
        """
        if self.SGD_Up:  # Stochastic Gradient Descent update
            error = np.dot(chosen_arm, self.hat_theta) - reward
            update = np.dot(chosen_arm, error) + self.lambda_reg * self.hat_theta
            self.hat_theta -= eta * update
        else:  # Least squares update
            self.cov += np.outer(chosen_arm, chosen_arm)
            self.invcov = pinv(self.cov)  # Update inverse covariance matrix
            self.b_t += reward * chosen_arm
            self.hat_theta = np.inner(self.invcov, self.b_t)  # Update parameter estimates
        self.t += 1  # Increment time step

    def name(self):
        """
        Returns the name of the algorithm, including configuration details.

        Returns:
            str: The name of the algorithm, with details of epsilon and the update method.
        """
        if self.SGD_Up:
            return f'LinESGD(eps = {self.eps}, m = {self.m})'
        else:
            return f'LinEGreedy({self.eps}, m = {self.m})'
