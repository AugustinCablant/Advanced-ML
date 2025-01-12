import numpy as np 

class RegretBalancingAgent:
    """Regret Bound Balancing and Elimination Algorithm."""
    def __init__(self, learners, K, delta, u_bounds, c, env):
        self.learners = learners  # list of base learners
        self.nb_learners = len(learners)  # number of base learners
        self.K = K  # number of arms
        self.N = None
        self.R = None
        self.G = None
        self.u_bounds = u_bounds  # Each base learner i comes with a candidate regret bound u_i(t,K) denoted by Ri in the paper
        self.env = env
        self.reset()

    def reset(self):
        self.ActiveLearners = self.learners  # set of active learners
        self.lenActiveLearners = len(self.ActiveLearners)  # number of active learners
        self.N = np.ones(self.nb_learners)  # number of rounds base learner i has been played at round t 
        self.R =  np.zeros(self.nb_learners) # total reward accumulated by base learner i after a total of t
        self.G = np.zeros(self.nb_learners)
        self.t = 0  # total number of rounds
        

    def get_action(self, action_set, t):
        set_to_maximize_index = [(self.R[i] / self.N[i]) + (self.u_bounds[i](self.t) / self.N[i]) for i in range(self.nb_learners)]
        j_t = np.argmax(set_to_maximize_index)
        b_t = self.R[j_t] / (self.N[j_t] + 10 ** (-6)) + self.u_bounds[j_t](self.t) / (self.N[j_t] + 10 ** (-6))
        # Empirical regret of base i at round t:
        self.G = self.N * b_t - self.R
        i_t = np.argmin(self.G)

        # play learner i_t 
        action = self.ActiveLearners[i_t].get_action(action_set, t)
        reward = self.env.get_reward(action)
        return i_t, action, reward
    
    def receive_reward(self, i_t, action, reward):
        self.ActiveLearners[i_t].receive_reward(action, reward)
        self.R[i_t] += reward
        self.N[i_t] += 1
        self.t += 1
        return None

    def name(self):
        return 'RB' 


# Try to implement the elimination strategy
class RegretBalancingAgentWithElimination:
    """Regret Bound Balancing and Elimination Algorithm."""
    def __init__(self, learners, K, delta, u_bounds, c, env):
        self.learners = learners  # list of base learners
        self.nb_learners = len(learners)  # number of base learners
        self.K = K  # number of arms
        self.c = c # constant c in the regret bound
        self.delta = delta 
        self.counts_learners = None
        self.U = None
        self.u_bounds = u_bounds  # Each base learner i comes with a candidate regret bound u_i(t,K) denoted by Ri in the paper
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
        action = self.ActiveLearners[i_t].get_action(action_set, t)
        reward = self.env.get_reward(action)
        self.ActiveLearners[i_t].receive_reward(action, reward)

        # update n_i and U_i
        self.counts_learners[i_t] += 1
        self.U[i_t] += reward

        index_to_remove = []
        # update active learners
        for i in range(self.lenActiveLearners):
            left_side = (self.U[i_t] / self.counts_learners[i]) + (self.u_bounds[i](self.counts_learners[i]) / self.counts_learners[i]) 
            sqrt_term = self.c * np.sqrt( np.log((np.log(self.counts_learners[i]) * self.nb_learners) / self.delta) / self.counts_learners[i])
            left_side += sqrt_term
            right_side = max([(self.U[j] / self.counts_learners[j]) - self.c * np.sqrt(np.log((np.log(self.counts_learners[j]) * self.nb_learners) / self.delta) / self.counts_learners[j]) for j in range(self.lenActiveLearners)]) 
            if left_side < right_side:
                index_to_remove.append(i)
                self.lenActiveLearners -= 1
            
            """ 
            if self.lenActiveLearners == 1:
                break
            """
        print(index_to_remove)
        return action, i_t


    def name(self):
        return 'RB with elimination' 