import numpy as np 

class RegretBalancing_Representation:
    """Define the regret balancing method for representation learning of linear mapping. Take as input LinUCB base learners."""
    def __init__(self, baseAlgo, K, u_bound, env):
        self.K = K
        self.baseAlgo = baseAlgo
        self.number_agents = len(baseAlgo)
        self.u_bound = lambda t,K : u_bound(t, K)
        self.env = env
        self.reset()

    def reset(self):
        self.counts = np.zeros(self.K)  # Assuming 4 arms
        self.rewards = np.zeros(self.K)
        self.rewards_per_agent = np.zeros(self.number_agents)
        self.t = 0

    def get_action(self, arms):
        # Regret balancing strategy: balancing reward and exploration term

        learners = self.baseAlgo
        env = self.env

        M = len(learners)

        nbrPlayed = np.zeros(M)
        rewards = np.zeros(M)
        potentialValues = np.zeros(M)

        for i in range(M):

            action = learners[i].get_action(arms) #Need to be updated depending on type of base learners

            nbrPlayed[i] = learners[i].get_numberPlayed()
            rewards[i] = env.get_reward(action)
            potentialValues[i] = (rewards[i] + self.u_bound(self.t,self.K))/(nbrPlayed[i] + 1e-6)

        
        b = np.max(potentialValues)
        empiricalRegret = np.zeros(M)

        for i in range(M):
            empiricalRegret[i] = nbrPlayed[i]*b - rewards[i]

        chosenLearner_idx = np.argmin(empiricalRegret)
        chosenLearner = learners[chosenLearner_idx]
        chosenAction  = chosenLearner.get_action(arms)

        return chosenAction, chosenLearner_idx
  

    def receive_reward(self, learner_idx, action, reward): #Need to include the update of choosen learner
        self.baseAlgo[learner_idx].receive_reward(action, reward)
        self.rewards_per_agent[learner_idx] += reward
        self.counts[learner_idx] += 1
        return None

    def name(self):
        return 'RB' 