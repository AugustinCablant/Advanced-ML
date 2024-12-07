import numpy as np

def ActionsGenerator(K,d, mean=None):
    """
    K: int -- number of action vectors to be generated
    d : int -- dimension of the action space
    returns : an array of K vectors uniformly sampled on the unit sphere in R^d
    """
    actions = np.random.randn(K, d)
    actions /= np.linalg.norm(actions, axis=1, keepdims=True)
    return actions