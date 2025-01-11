import numpy as np 

class RegretBalancingAgent:
    """Regret Bound Balancing and Elimination Algorithm."""
    def __init__(self, learners, K, delta, u_bounds, c, env):
        self.learners = learners  # list of base learners
        self.nb_learners = len(learners)  # number of base learners
        self.K = K  # number of arms
        self.c = c # constant c in the regret bound
        self.delta = delta 
        self.counts_learners = None
        self.U = None
        self.u_bounds = [lambda x : u_bound(x) for u_bound in u_bounds]   # Each base learner i comes with a candidate regret bound u_i(t,K) denoted by Ri in the paper
        self.env = env
        self.reset()

    def reset(self):
        self.ActiveLearners = self.learners  # set of active learners
        self.lenActiveLearners = len(self.ActiveLearners)  # number of active learners
        self.U =  np.zeros(self.nb_learners) # total reward accumulated by base learner i after a total of t
        self.t = 0  # total number of rounds
        self.counts_learners = np.zeros(self.nb_learners)  # number of rounds base learner i has been played (n_i(t))
        

    def get_action(self, action_set, t):
        # Regret balancing strategy: balancing reward and exploration term
        i_t = np.argmin([bound(self.counts_learners[i]) for i, bound in enumerate(self.u_bounds)])

        # play learner i_t and receive corresponding reward
        action = self.ActiveLearners[i_t].get_action(action_set)
        reward = self.env.get_reward(action)
        self.ActiveLearners[i_t].receive_reward(action, reward)

        # update n_i and U_i
        self.counts_learners[i_t] += 1
        self.U[i_t] += reward

        # update active learners
        for i in range(self.lenActiveLearners):
            left_side = (self.U[i_t] / self.counts_learners[i]) + (self.u_bounds[i](self.counts_learners[i]) / self.counts_learners[i]) 
            sqrt_term = self.c * np.sqrt( np.log((np.log(self.counts_learners[i]) * self.nb_learners) / self.delta) / self.counts_learners[i])
            left_side += sqrt_term
            right_side = max([(self.U[j] / self.counts_learners[j]) - self.c * np.sqrt(np.log((np.log(self.counts_learners[j]) * self.nb_learners) / self.delta) / self.counts_learners[j]) for j in range(self.lenActiveLearners)]) 
            if left_side < right_side:
                self.ActiveLearners.pop(i)
                self.lenActiveLearners -= 1
        return action, i_t


    def name(self):
        return 'RB' 