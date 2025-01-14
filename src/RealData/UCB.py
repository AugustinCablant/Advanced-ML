import numpy as np 
import matplotlib.pyplot as plt

def ucb_policy(df, t, slate_size=5, batch_size=50, ucb_scale=2.0):
    """
    Applies the Upper Confidence Bound (UCB) policy to generate movie recommendations.

    The UCB policy is a popular method in multi-armed bandit problems. It balances exploration 
    (trying underexplored options) and exploitation (choosing options with the highest observed rewards) 
    by assigning a confidence bound to each action. This confidence bound increases with uncertainty 
    about an action's mean reward and decreases as more information about the action is gathered.

    Args:
        df (pd.DataFrame): A DataFrame containing the historical data of movie interactions.
            Must include the following columns:
                - 'movieId' (int or str): The identifier of each movie (arm).
                - 'liked' (float or int): The reward signal, e.g., whether the movie was liked.
        t (int): The current time step, used to calculate the exploration term. It should be 
            greater than 0 and represent the number of rounds or batches processed so far.
        slate_size (int, optional): The number of recommendations to generate in each step. 
            Defaults to 5.
        batch_size (int, optional): The number of users served recommendations in each batch. 
            This parameter is not directly used in the function but can influence how 
            `t` is defined in the context of the broader system. Defaults to 50.
        ucb_scale (float, optional): A scaling factor for the exploration term. Most implementations 
            use a value of 2.0, but it can be adjusted to control the balance between exploration 
            and exploitation. Defaults to 2.0.

    Returns:
        recs (np.ndarray): A NumPy array containing the `movieId`s of the recommended movies, 
        ordered by their UCB scores in descending order.

    Notes:
        - The UCB score is computed as:
            UCB = mean_reward + sqrt((ucb_scale * log10(t)) / count)
          where:
            - `mean_reward` is the average reward for each movie.
            - `log10(t)` encourages exploration for movies with fewer interactions.
            - `count` is the number of times a movie has been selected.
        - Movies with no interactions (count = 0) should be handled in the calling logic if 
          they are part of the dataset but not included in the `df` used by this function.
    """
    # Calculate mean, count, and standard deviation of rewards for each movie
    scores = df[['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count', 'std']})
    scores.columns = ['mean', 'count', 'std']

    # Compute UCB scores
    scores['ucb'] = scores['mean'] + np.sqrt(
        (ucb_scale * np.log10(t)) / (scores['count'] + 1e-6))
    
    # Extract top recommendations based on UCB scores
    scores['movieId'] = scores.index
    scores = scores.sort_values('ucb', ascending=False)
    recs = scores.loc[scores.index[0:slate_size], 'movieId'].values
    return recs

def run_UCB_policy(df, history, slate_size=5, batch_size=50, ucb_scale=2.0, verbose=False, plot=False):
    """
    """
    # Initialize variables
    rewards = []
    max_time = df.shape[0]  # Total number of time steps (ratings) to evaluate

    # Iterate over batches
    for step in range(max_time // batch_size):
        t = step * batch_size

        # Print progress if verbose is enabled
        if verbose and t % 100000 == 0:
            print(f"Processing step {t}...")

        # Choose arms (recommendations) using epsilon-greedy policy
        recs = ucb_policy(df=history.loc[history.t<=t,], t = t/batch_size, ucb_scale=ucb_scale)

        # Score the recommendations and update history
        history, action_score = score(history, df, t, batch_size, recs)

        # Accumulate rewards from action scores
        if action_score is not None:
            rewards.extend(action_score.liked.tolist())
    if plot:
        plt.figure(figsize = (10, 5))
        plt.plot(np.cumsum(rewards), label = 'Cumulative Reward')
        plt.xlabel('Time Step')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward over Time Step for UCB Policy')
        plt.legend()
        plt.show()
    return rewards, history


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

