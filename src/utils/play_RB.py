import numpy as np 

def play_RB(environment, agent, Nmc, T):
    """
    Play one Nmc trajectories over a horizon T for the specified agent.
    Return the agent's name (sring) and the collected data in an nd-array.
    """

    data = np.zeros((Nmc, T))

    for n in range(Nmc):
        environment.reset()   
        agent.reset()

        for t in range(T):
            action_set = environment.get_action_set()
            #action = agent.get_action(action_set, t)
            i_t, action, reward = agent.get_action(action_set, t)
            reward = environment.get_reward(action)
            agent.receive_reward(i_t, action, reward) 
        
            # compute instant (pseudo) regret
            means = environment.get_means()
            best_reward_arm = np.max(means)
            data[n,t]= np.random.binomial(1, best_reward_arm) - reward 

    return agent.name(), data