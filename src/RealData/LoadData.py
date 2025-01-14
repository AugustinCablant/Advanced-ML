import pandas as pd 
import numpy as np 

def read_data_20m():
	print('reading movielens 20m data')
	ratings = pd.read_csv('/Users/augustincablant/Documents/GitHub/Advanced-ML/data/ml-25m/ratings.csv', engine='python')  # 'data/ml-25m/ratings.csv'
	movies = pd.read_csv('/Users/augustincablant/Documents/GitHub/Advanced-ML/data/ml-25m/movies.csv', engine='python')   # 'data/ml-25m/movies.csv'
	#links = pd.read_csv('data/ml-25m/links.csv', engine='python')
	#tags = pd.read_csv('data/ml-25m/tags.csv', engine='python')
	movies = movies.join(movies.genres.str.get_dummies().astype(bool))
	movies.drop('genres', inplace=True, axis=1)
	logs = ratings.join(movies, on='movieId', how='left', rsuffix='_movie')
	return logs


def preprocess_movie_data_20m(logs, min_number_of_reviews = 20000, balanced_classes = False):
	print('preparing ratings log')
	# remove ratings of movies with < N ratings. too few ratings will cause the recsys to get stuck in offline evaluation
	# Get counts of ratings for each movie
	movie_counts = logs['movieId'].value_counts()

	# Filter to keep only movies with enough ratings
	movies_to_keep = movie_counts[movie_counts >= min_number_of_reviews].index

	# Filter the logs to include only those movies
	logs = logs[logs['movieId'].isin(movies_to_keep)]
	logs = logs.loc[logs['movieId'].isin(movies_to_keep)]
	if balanced_classes is True:
		logs = logs.groupby('movieId')
		logs = logs.apply(lambda x: x.sample(logs.size().min()).reset_index(drop=True))
	# shuffle rows to deibas order of user ids
	logs = logs.sample(frac=1)
	# create a 't' column to represent time steps for the bandit to simulate a live learning scenario
	logs['t'] = np.arange(len(logs))
	logs.index = logs['t']
	logs['liked'] = logs['rating'].apply(lambda x: 1 if x >= 4.5 else 0)
	return logs

def get_ratings_20m(min_number_of_reviews=20000, balanced_classes=False):
	logs = read_data_20m()
	logs = preprocess_movie_data_20m(logs, min_number_of_reviews=20000, balanced_classes=balanced_classes)
	return logs

def create_dataframe(min_review_count = 1500, balanced_classes = False):
	return get_ratings_20m(min_number_of_reviews = min_review_count, 
						balanced_classes = balanced_classes)