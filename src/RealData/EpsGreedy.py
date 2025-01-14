import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def epsilon_greedy_policy(df, arms, epsilon=0.15, slate_size=5, batch_size=50):
    """
    Epsilon-Greedy Policy for Recommender Systems or Bandit Problems.

    This function implements an epsilon-greedy policy to decide which items 
    to recommend to users based on historical data of user interactions.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing user interactions with items. It should include:
        - 'movieId': Identifier for the items (e.g., movies).
        - 'liked': Indicator of whether the user liked the item (1 for like, 0 for dislike).
    arms : list
        The set of all possible items that can be recommended (e.g., a list of movie IDs).
    epsilon : float, optional, default=0.15
        The probability of exploring (choosing random items) rather than exploiting
        (choosing the best-performing items so far).
    slate_size : int, optional, default=5
        The number of recommendations to generate in each step.
    batch_size : int, optional, default=50
        The number of users to serve recommendations before potentially updating the policy.
        (Not directly used in this function but relevant in larger batch-processing contexts).

    Returns
    -------
    recs : numpy.ndarray
        An array of item IDs representing the recommendations for the current step.

    Policy Logic
    ------------
    - With a probability of `epsilon`, the policy chooses to explore:
      - A random set of `slate_size` items is selected from `arms` without replacement.
    - Otherwise (with a probability of `1 - epsilon`), the policy chooses to exploit:
      - Items are ranked by their average like rate (`liked` column in `df`).
      - The top `slate_size` items with the highest mean like rates are recommended.
    - If `df` is empty, the policy defaults to exploration.
    """
    # Draw a 0 or 1 from a binomial distribution, with epsilon likelihood of drawing a 1
    explore = np.random.binomial(1, min(1, epsilon))
    
    # If exploring, shuffle items to choose a random set of recommendations
    if explore == 1 or df.shape[0] == 0:
        recs = np.random.choice(arms, size=(slate_size), replace=False)

    # If exploiting, sort items by "like rate" and recommend the best-performing items
    else:
        scores = df[['movieId', 'liked']].groupby('movieId').agg({'liked': ['mean', 'count']})
        scores.columns = ['mean', 'count']
        scores['movieId'] = scores.index
        scores = scores.sort_values('mean', ascending=False)
        recs = scores.loc[scores.index[0:slate_size], 'movieId'].values

    return recs



def run_epsilon_policy(df, history, epsilon, slate_size, batch_size, verbose=False, plot=False, fixed_epsilon=True):
    """
    Runs an epsilon-greedy bandit policy on a dataset, evaluating recommendations 
    and accumulating rewards over multiple time steps.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing logged user interactions, with the following columns:
        - 'movieId': Identifier for items.
        - 'liked': Indicator of whether the item was liked (1 for like, 0 for dislike).
    history : pandas.DataFrame
        The history dataframe to track actions and their scores. Should have the same structure as `df`.
    epsilon : float
        The probability of exploring random arms (recommendations) instead of exploiting the best-known arms.
    n : int
        The number of recommendations (slate size) to make at each time step.
    batch_size : int
        The number of users to evaluate recommendations for in each batch.
    verbose : bool, optional (default=False)
        If True, prints progress every 100,000 time steps.

    Returns
    -------
    rewards : list
        A list of rewards (1 for liked items, 0 for disliked items) accumulated over all time steps.
    updated_history : pandas.DataFrame
        The updated history dataframe containing all actions and scores.
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

        if fixed_epsilon==False:
            if t == 0:
                epsilon_ = epsilon 
            else:
                epsilon_ = epsilon / t

        # Choose arms (recommendations) using epsilon-greedy policy
        recs = epsilon_greedy_policy(df=history.loc[history.t<=t,], arms=df.movieId.unique(), epsilon=epsilon_, slate_size=slate_size, batch_size=batch_size)

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
        plt.title('Cumulative Reward over Time Step for Epsilon Greedy Policy with Epsilon = {}'.format(epsilon))
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

