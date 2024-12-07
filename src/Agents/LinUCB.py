from src.utils.imports import * 

class LinUCB:
    def __init__(self, d, lambda_reg, delta = 0.1, sigma = 1., L = 1.):
        self.d = d
        self.lambda_reg = lambda_reg
        self.delta = delta
        self.sigma = sigma
        self.L = L
        self.reset()

    def reset(self):
        self.t = 0
        self.hat_theta = np.zeros(self.d)
        self.cov = self.lambda_reg * np.identity(self.d)
        self.invcov = (1 / self.lambda_reg) * np.identity(self.d)
        self.b_t = np.zeros(self.d)

    def beta(self):
        return self.sigma * np.sqrt(
            2 * np.log(1 / self.delta) + self.d * np.log(1 + self.t * ((self.L) / (self.d * self.lambda_reg)))
            ) + np.sqrt(self.lambda_reg) * norm(self.hat_theta)

    def UCB(self, a):
        self.b_t = self.beta()
        return np.dot(a, self.hat_theta) + self.b_t * np.sqrt(np.dot(a, np.dot(self.invcov, a)))

    def get_action(self, arms):
        K, _ = arms.shape
        UCB = np.zeros(K)
        for k in range(K):
            a = arms[k]
            UCB[k] = self.UCB(a)
        A_t = arms[np.argmax(UCB)]
        return A_t
    
    def receive_reward(self, chosen_arm, reward):
        self.cov += np.outer(chosen_arm, chosen_arm)
        self.invcov = pinv(self.cov)    # update inverse covariance matrix
        self.b_t += reward * chosen_arm
        self.hat_theta = np.inner(self.invcov, self.b_t)    # update the least square estimate
        self.t += 1
    
    def name(self):
        return 'LinUCB'