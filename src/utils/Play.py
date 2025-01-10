import numpy as np 

def play(environment, agent, Nmc, T):
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
                if agent.name().split('(')[0] == 'EpsilonGreedy':
                    action = agent.get_action(action_set, t)
                elif agent.name().split('(')[0] == 'RB':
                    action, learner = agent.get_action(action_set)
                else:
                    action = agent.get_action(action_set)

            reward = environment.get_reward(action)
            if agent.name().split('(')[0] == 'RB':
                agent.receive_reward(learner, action, reward)
            else: 
                agent.receive_reward(action, reward)

            # compute instant (pseudo) regret
            means = environment.get_means()
            best_reward_arm = np.max(means)
            data[n,t]= np.random.binomial(1, best_reward_arm) - reward 

    return agent.name(), data