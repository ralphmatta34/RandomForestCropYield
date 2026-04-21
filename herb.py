import random
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score


def clean_herb_data_df(df):
	"""
	Clean up the dataframe

	Parameters:
	df (pandas df): the dataframe.

	Returns:
	tuple: 2 dictionaries mapping items and areas to their encodings.
	"""
	
	items = {}
	for item in full_df['Item']:
		if item in items:
			items[item] += 1
		elif item not in items:
			items[item] = 1
	
	item_codes = {}
	for i in range(len(items.keys())):
		item = list(items.keys())[i]
		item_codes[item] = i
	
	full_df['Encoded_item'] = [item_codes[item] for item in full_df['Item']]
	
	areas = set()
	for area in full_df['Area']:
		if area not in areas:
			areas.add(area)
	
	area_codes = {}
	codes_to_areas = {}
	
	for index, area in enumerate(areas):
		area_codes[area] = index
		codes_to_areas[index] = area
	
	full_df['Encoded_area'] = [area_codes[area] for area in full_df['Area']]
	
	full_df.to_csv('yield_df.csv', index=False)
	
	return item_codes, area_codes


full_df = pd.read_csv('yield_df.csv')
encodings = clean_herb_data_df(full_df)


def get_herb_feature_columns():
	"""
	Access the feature columns from our dataset.

	Returns:
	array: the "independent variable" columns from the dataset
	"""
	
	X_columns = ['Encoded_item', 'Year', 'Encoded_area'] # TODO: Can change features used
	
	return X_columns


def get_herb_target():
	"""
	Access the target column from our dataset.

	Returns:
	array: the "dependent variable" column from the dataset.
	"""
	
	y_columns = ['hg/ha_yield']
	
	return y_columns


def predict_herb_yield(produce, year, area, trained_model):
	"""
	Make an inference of crop yield.

	Parameters:
	produce (str): the type of crop.
	year (int): the year of measurement.
	area (str): the country of measurement.
	trained_model (sklearn model): the trained model with which to make the inference.

	Returns:
	dict: produce, year, and corresponding prediction of yield.
	"""
	
	X = [[encodings[0][produce], year, encodings[1][area]]]
	X = pd.DataFrame(X, columns=['Encoded_item', 'Year', 'Encoded_area'])
	y_pred = trained_model.predict(X)
	
	dict = {"produce": produce, "year": year, "yield": y_pred}
	
	return dict  # TODO: add most most relevant features by feature importances


def predict_herb_optimal_location(produce, year, trained_model):
	"""
	Optimise the location for growing a given type of crop.

	Parameters:
	produce (str): the type of crop.
	year (int): the year of measurement.
	trained_model (sklearn model): the trained model with which to make the prediction.

	Returns:
	string: optimal location
	"""
	
	yields = []
	areas = set(encodings[1].keys())
	for area in areas:
		predicted_yield = predict_herb_yield(produce, 2024, area)
		yields.append([predicted_yield, area])
	
	location = max(yields, key=lambda x: x[0]["yield"])[1]
	
	return location
