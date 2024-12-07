from src.utils.imports import *
from src.utils.imports import ActionGenerator as ActionsGenerator

class LinearBandit:

    def __init__(self, theta, K, var=1., fixed_actions = None):
      """
      theta: d-dimensional vector (bounded) representing the hidden parameter
      K: number of actions per round (random action vectors generated each time)
      pb_type: string in 'fixed', 'iid', 'nsr' (please ignore NotSoRandom)
      """
      self.d = np.size(theta)
      self.theta = theta
      self.K = K
      self.var = var
      self.current_action_set = np.zeros(self.d)
      self.fixed_actions = fixed_actions


    def get_action_set(self):
      """
      Generates a set of vectors in dimension self.d
      """
      if self.fixed_actions is None or not self.fixed_actions.size:
        self.current_action_set = ActionsGenerator(self.K, self.d)
      else:
        self.current_action_set = self.fixed_actions
      return self.current_action_set


    def get_reward(self, action):
      """ sample reward given action and the model of this bandit environment
      action: d-dimensional vector (action chosen by the learner)
      """
      mean = np.dot(action, self.theta)
      return np.random.normal(mean, scale=self.var)

    def get_means(self):
      return np.dot(self.current_action_set, self.theta)
    
    def reset(self):
        """
        Reset the environment for a new simulation or episode.
        If the environment is IID, regenerate the action set.
        """
        if self.fixed_actions is None:
            # Regenerate the action set if IID (random actions at each round)
            self.current_action_set = ActionsGenerator(self.K, self.d)
        else:
            # If fixed actions, just reset to the predefined set
            self.current_action_set = self.fixed_actions