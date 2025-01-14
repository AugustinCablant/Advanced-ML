import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def RegretBalancingEpsilonGreedy(df, arms, learners, slate_size=5, batch_size=50):















def score(history, df, t, batch_size, recs):
    """
    Calculate the replay score for a batch of recommendations.

    This function evaluates the recommendations (`recs`) by checking if they match 
    the logged data in the dataset `df` for a given time step `t`. Only the items 
    that match the logged data are scored, while others are ignored.

    Parameters
    ----------
    history : pandas.DataFrame
        The cumulative history of actions (logged interactions with users) that 
        match the recommendations. New matching actions will be appended to this 
        dataframe.
    df : pandas.DataFrame
        The dataset containing logged interactions with the following expected columns:
        - 'movieId': Identifier for the recommended items.
        - 'liked': Indicator of whether the user liked the item (1 for like, 0 for dislike).
    t : int
        The current time step (used to determine which portion of `df` to evaluate).
    batch_size : int
        The number of users to evaluate recommendations for in the current batch.
    recs : list or numpy.ndarray
        The list of recommended item IDs for evaluation.

    Returns
    -------
    history : pandas.DataFrame
        The updated history dataframe with rows from `df` that matched the recommendations.
    action_liked : pandas.DataFrame
        A dataframe containing the items from `recs` that were liked by users, 
        with columns:
        - 'movieId': Identifier for the items.
        - 'liked': Indicator of whether the item was liked (1 for like, 0 for dislike).

    Details
    -------
    - This function implements a "replay score" as described in the referenced paper:
      https://arxiv.org/pdf/1003.5956.pdf.
    - Recommendations are rewarded only if they match the items logged in the dataset (`df`).
    - Items that do not appear in the logged data are ignored.
    """
    # Select the actions (logged data) for the current batch
    actions = df[t:t+batch_size]

    # Filter actions that match the recommended items
    actions = actions.loc[actions['movieId'].isin(recs)]

    # Mark the scoring round for the matched actions
    actions['scoring_round'] = t

    # Append the matched actions to the history
    history = pd.concat([history, actions])

    # Extract the 'movieId' and 'liked' columns for evaluation
    action_liked = actions[['movieId', 'liked']]

    return history, action_liked
