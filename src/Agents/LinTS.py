!cd "/Users/augustincablant/Documents/GitHub/Advanced-ML"
from src.utils.imports import *

class LinTS:
    """
    Implementation of the Linear Thompson Sampling (LinTS) algorithm for contextual bandits.
    This algorithm uses Bayesian inference to balance exploration and exploitation by 
    sampling model parameters from a posterior distribution.

    Attributes:
        d (int): Dimensionality of the feature space.
        lambda_reg (float): Regularization parameter for the prior covariance matrix.
        sigma (float): Noise standard deviation for the reward distribution (default is 1.0).

    Methods:
        reset():
            Resets the algorithm's internal state, including the prior mean, covariance, 
            and sampled parameter vector.

        update_theta(chosen_arm, reward):
            Updates the posterior mean and covariance based on the observed reward and 
            the chosen action, and samples a new parameter vector.

        get_action(arms):
            Selects an action by computing the expected reward for each arm based on 
            the sampled parameter vector and returning the arm with the highest expected reward.

        receive_reward(chosen_arm, reward):
            Updates the model parameters based on the chosen arm and the observed reward, 
            and increments the time step.

        name():
            Returns the name of the algorithm, "LinTS".
    """

    def __init__(self, d, lambda_reg, sigma=1.0):
        """
        Initializes the LinTS algorithm.

        Args:
            d (int): Dimensionality of the feature space.
            lambda_reg (float): Regularization parameter for the prior covariance matrix.
            sigma (float): Noise standard deviation for the reward distribution (default is 1.0).
        """
        self.d = d
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        self.reset()

    def reset(self):
        """
        Resets the internal state of the algorithm.
        Reinitializes the prior mean, covariance matrix, and sampled parameter vector.
        """
        self.t = 0  # Time step
        self.cov = self.lambda_reg * np.identity(self.d)  # Prior covariance matrix
        self.mean = np.zeros(self.d)  # Prior mean
        self.hat_theta = np.random.multivariate_normal(self.mean, self.cov)  # Sampled parameter vector (prior)

    def get_numberPlayed(self):
        """ Return number of times this agent has been played. """
        return self.t
    
    def update_theta(self, chosen_arm, reward):
        """
        Updates the posterior mean and covariance based on the chosen action and observed reward.
        Samples a new parameter vector from the updated posterior distribution.

        Args:
            chosen_arm (numpy.ndarray): The feature vector of the chosen action.
            reward (float): The observed reward for the chosen action.
        """
        previous_cov = self.cov
        B_t = pinv(previous_cov / self.sigma)  # Precision matrix of the prior
        previous_sum = B_t @ self.mean  # Weighted sum of prior contributions
        B_t += np.outer(chosen_arm, chosen_arm)  # Update precision matrix with new action
        inv_B = pinv(B_t)  # Compute posterior covariance
        self.cov = inv_B * self.sigma  # Update posterior covariance
        new_sum = previous_sum + reward * chosen_arm  # Update weighted sum with new reward
        self.mean = inv_B @ new_sum  # Update posterior mean
        self.hat_theta = np.random.multivariate_normal(self.mean, self.cov)  # Sample new parameter vector

    def get_action(self, arms):
        """
        Selects an action by maximizing the expected reward based on the sampled parameter vector.

        Args:
            arms (numpy.ndarray): A 2D array where each row represents an action's feature vector.

        Returns:
            numpy.ndarray: The feature vector of the selected action.
        """
        K, _ = arms.shape
        vector = [np.dot(action, self.hat_theta) for action in arms]  # Compute expected rewards
        chosen_arm = arms[np.argmax(vector)]  # Select the action with the highest expected reward
        return chosen_arm

    def receive_reward(self, chosen_arm, reward):
        """
        Updates the model parameters based on the chosen action and observed reward.
        Increments the time step.

        Args:
            chosen_arm (numpy.ndarray): The feature vector of the chosen action.
            reward (float): The observed reward for the chosen action.
        """
        reward = np.dot(chosen_arm, self.hat_theta)  # Update the reward for the chosen action
        self.update_theta(chosen_arm, reward)  # Update the posterior parameters
        self.t += 1  # Increment time step

    def name(self):
        """
        Returns the name of the algorithm.

        Returns:
            str: The name "LinTS".
        """
        return "LinTS"
