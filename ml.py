import random
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score


def hyperparameter_search(df, model):
	"""
	Find the best hyperparameters and weights.

	Parameters:
	df (pandas df): the dataframe with crop data.
	model (sklearn model): the untrained model.

	Returns:
	optimised_model: the optimised model
	"""
	
	param_grid = {
		'n_estimators': [10, 25, 50, 100, 175],
		'max_depth': [None, 10, 20, 30],
		'min_samples_split': [2, 5, 10],
		'min_samples_leaf': [1, 2, 4],
		'max_features': ['auto', 'sqrt', 'log2', None]
	}
	
	random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10,
					scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
	X = df[get_herb_feature_columns()]
	y = df[get_herb_target()]
	random_search.fit(X, y)
	best_hyperparams = random_search.best_params_
	
	optimised_model = RandomForestRegressor(
		n_estimators=best_hyperparams['n_estimators'],
		max_depth=best_hyperparams['max_depth'],
		min_samples_split=best_hyperparams['min_samples_split'],
		min_samples_leaf=best_hyperparams['min_samples_leaf'],
		max_features=best_hyperparams['max_features']
	)
	
	return optimised_model, best_hyperparams

def optimise(df, model):
	"""
	Hyperparameter tune the model.

	Parameters:
	df (pandas df): the dataframe with crop data.
	model (sklearn model): the untrained model.

	Returns:
	optimised_model: the optimised model
	"""
	
	optimised_model = hyperparameter_search(RandomForestRegressor)[0]
	
	return optimised_model
