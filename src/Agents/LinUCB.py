from utils.imports import * 

class LinUCB:
    """
    Implementation of the Linear Upper Confidence Bound (LinUCB) algorithm for contextual bandits.
    This algorithm balances exploration and exploitation by maintaining a confidence interval
    around the estimated parameters for each action.

    Attributes:
        d (int): Dimensionality of the feature space.
        lambda_reg (float): Regularization parameter for the ridge regression estimator.
        delta (float): Confidence level for the UCB (default is 0.1).
        sigma (float): Standard deviation of the noise in the rewards (default is 1.0).
        L (float): Upper bound on the norm of feature vectors (default is 1.0).

    Methods:
        reset():
            Resets the internal state of the algorithm, including the time step, 
            parameter estimates, and covariance matrices.

        beta():
            Computes the exploration bonus term based on the confidence level 
            and current time step.

        UCB(a):
            Calculates the Upper Confidence Bound (UCB) value for a given action.

        get_action(arms):
            Selects the action with the highest UCB value from a set of arms.

        receive_reward(chosen_arm, reward):
            Updates the parameter estimates and covariance matrices based on the 
            observed reward and the chosen arm.

        name():
            Returns the name of the algorithm ("LinUCB").
    """

    def __init__(self, d, lambda_reg, delta=0.1, sigma=1., L=1., prefactor = 1.):
        """
        Initializes the LinUCB algorithm.

        Args:
            d (int): Dimensionality of the feature space.
            lambda_reg (float): Regularization parameter for ridge regression.
            delta (float): Confidence level for the UCB (default is 0.1).
            sigma (float): Standard deviation of noise in rewards (default is 1.0).
            L (float): Upper bound on the norm of feature vectors (default is 1.0).
        """
        self.d = d
        self.lambda_reg = lambda_reg
        self.delta = delta
        self.sigma = sigma
        self.L = L
        self.prefactor = prefactor
        self.reset()

    def reset(self):
        """
        Resets the internal state of the algorithm.
        Initializes the time step, parameter estimates, and covariance matrices.
        """
        self.t = 0
        self.hat_theta = np.zeros(self.d)  # Estimated parameters
        self.cov = self.lambda_reg * np.identity(self.d)  # Covariance matrix
        self.invcov = np.identity(self.d)  # Inverse covariance
        self.b_t = np.zeros(self.d)  # Accumulated reward-weighted features

    def get_numberPlayed(self):
        """ Return number of times this agent has been played. """
        return self.t
    
    def beta(self):
        """
        Computes the exploration bonus term.

        Returns:
            float: The exploration bonus based on confidence level and time step.
        """
        return self.sigma * np.sqrt(
            2 * np.log(1 / self.delta) + self.d * np.log(1 + self.t * (self.L / (self.d * self.lambda_reg)))
        ) + np.sqrt(self.lambda_reg)

    def UCB(self, a):
        """
        Calculates the Upper Confidence Bound (UCB) for a given action.

        Args:
            a (numpy.ndarray): Feature vector of the action.

        Returns:
            float: The UCB value for the action.
        """
        beta = self.beta()
        return self.prefactor*np.dot(a, self.hat_theta) + np.sqrt(np.dot(a, self.invcov@a)) * beta

    def get_action(self, arms):
        """
        Selects the action with the highest UCB value from a set of arms.

        Args:
            arms (numpy.ndarray): A 2D array where each row represents an action's feature vector.

        Returns:
            numpy.ndarray: The feature vector of the selected action.
        """
        index = np.argmax([self.UCB(arm) for arm in arms])
        action = arms[index]
        return action

    def receive_reward(self, chosen_arm, reward):
        """
        Updates the model parameters based on the observed reward.

        Args:
            chosen_arm (numpy.ndarray): The feature vector of the chosen action.
            reward (float): The observed reward for the chosen action.
        """
        self.cov += np.outer(chosen_arm, chosen_arm)
        self.invcov = pinv(self.cov)  # Update inverse covariance matrix
        self.b_t += reward * chosen_arm
        self.hat_theta = np.inner(self.invcov, self.b_t)  # Update parameter estimates
        self.t += 1  # Increment time step

    def name(self):
        """
        Returns the name of the algorithm.

        Returns:
            str: The name "LinUCB".
        """
        return f'LinUCB({self.prefactor})'
