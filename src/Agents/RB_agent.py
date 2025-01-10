import numpy as np 

class RegretBalancingAgent:
    def __init__(self, K, delta, u_bound):
        self.K = K
        self.delta = delta
        self.counts = None
        self.rewards = None
        self.u_bound = lambda x, d: u_bound(x, d)
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.K)  # Assuming 4 arms
        self.rewards = np.zeros(self.K)
        self.t = 0

    def get_action(self, action_set, t):
        # Regret balancing strategy: balancing reward and exploration term
        u_t = self.u_bound(t, self.delta)
        confidence_bounds = self.rewards / (self.counts + 1e-5) + u_t / (self.counts + 1e-5)
        return np.argmax(confidence_bounds)

    def receive_reward(self, action, reward):
        self.counts[action] += 1
        self.rewards[action] += reward

    def name(self):
        return 'RB( delta = '+str(self.delta)+')' 