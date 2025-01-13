from utils.Play import play
from utils.play_RB import play_RB

def experiment(environment, agents, Nmc, T, bern = False):
    """
    Play Nmc trajectories for all agents over a horizon T. Store all the data in a dictionary.
    """

    all_data = {}

    for agent in agents:
        if agent.name() == 'RB':
            agent_id, regrets = play_RB(environment, agent, Nmc, T, bern=bern)
        else:
            agent_id, regrets = play(environment, agent, Nmc, T, bern=bern)

        all_data[agent_id] = regrets

    return all_data