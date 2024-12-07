from src.utils.imports import *

class LinEpsilonGreedy:
  def __init__(self, d, lambda_reg,m = 1, eps = 0.1, SGD_Up = False, FixedEpsilon = False, other_option = None):
    self.eps = eps # exploration probability
    self.d = d
    self.lambda_reg = lambda_reg
    self.SGD_Up = SGD_Up
    self.m = m
    self.FixedEpsilon = FixedEpsilon
    self.reset()

  def reset(self):
    """
    This function should reset all estimators and counts.
    It is used between independent experiments (see 'Play!' above)
    """
    self.t = 0
    self.hat_theta = np.ones(self.d)
    self.cov = self.lambda_reg * np.identity(self.d)
    self.invcov = np.identity(self.d)
    self.b_t = np.zeros(self.d)

  def get_action(self, arms, t):
    K, _ = arms.shape
    if t < self.m * K:     # play each arm once (exploration)
      index = t % K
      return arms[index]
    else:
      expl = np.random.random()
      if self.FixedEpsilon:
        self.eps = np.min(1, (K) / (self.t * (self.d ** 2)))
      if expl < self.eps:
        action = arms[np.random.randint(K)]
      else:
        index = np.argmax([np.dot(arm, self.hat_theta) for arm in arms])
        action = arms[index]
      return action

  def receive_reward(self, chosen_arm, reward, eta = 0.01):
    """
    update the internal quantities required to estimate the parameter theta using least squares
    """
    if self.SGD_Up:
      error = np.dot(chosen_arm, self.hat_theta) - reward
      update = np.dot(chosen_arm, error) + self.lambda_reg * self.hat_theta
      self.hat_theta -= eta * update
    else:
      #update inverse covariance matrix
      self.cov += np.outer(chosen_arm, chosen_arm)
      self.invcov = pinv(self.cov)


      #update b_t
      self.b_t += reward * chosen_arm

      self.hat_theta = np.inner(self.invcov, self.b_t) # update the least square estimate
    self.t += 1

  def name(self):
    if self.SGD_Up:
      return 'LinESGD(eps = '+str(self.eps)+', m = '+str(self.m)+')'
    else:
      return 'LinEGreedy('+str(self.eps)+', m = '+str(self.m)+')'