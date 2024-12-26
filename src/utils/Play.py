import numpy as np 

def play(environment, agent, Nmc, T, pseudo_regret=True):
    """
    Play one Nmc trajectories over a horizon T for the specified agent.
    Return the agent's name (sring) and the collected data in an nd-array.
    """

    data = np.zeros((Nmc, T))

    for n in range(Nmc):
        environment.reset()    # if the environment is iid, reset it ; otherwise we have the same action set
        agent.reset()
        parameter_T = False

        if agent.name().split('(')[0] in ['LinESGD', 'LinEGreedy']:
           parameter_T = True

        for t in range(T):
            action_set = environment.get_action_set()

            if parameter_T:
               action = agent.get_action(action_set, t)
            else:
                action = agent.get_action(action_set)
            reward = environment.get_reward(action)
            agent.receive_reward(action,reward)

            # compute instant (pseudo) regret
            means = environment.get_means()
            best_reward = np.max(means)
            if pseudo_regret:
              # pseudo-regret removes some of the noise and corresponds to the metric studied in class
              data[n,t] = best_reward - np.dot(environment.theta,action)
            else:
              data[n,t]= best_reward - reward # this can be negative due to the noise, but on average it's positive

    return agent.name(), data