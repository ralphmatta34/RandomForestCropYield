import random
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score

# Cleaning up data and numerically encoding categorical data

full_df = pd.read_csv('yield_df.csv')
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


# Hyperparameter optimisation
# RandomForestRegressor was the best-performing model

def save_weights(model):
	param_grid = {
    	'n_estimators': [10, 25, 50, 100, 175],
    	'max_depth': [None, 10, 20, 30],
    	'min_samples_split': [2, 5, 10],
    	'min_samples_leaf': [1, 2, 4],
    	'max_features': ['auto', 'sqrt', 'log2', None]
	}

	random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
	random_search.fit(X_train, y_train)
	best_hyperparams = random_search.best_params_
	
	optimised_model = RandomForestRegressor(
    	n_estimators=best_hyperparams['n_estimators'],
    	max_depth=best_hyperparams['max_depth'],
    	min_samples_split=best_hyperparams['min_samples_split'],
    	min_samples_leaf=best_hyperparams['min_samples_leaf'],
    	max_features=best_hyperparams['max_features']
	)

	return optimised_model, best_hyperparams

X_columns = ['Encoded_item', 'Year', 'Encoded_area'] # TODO: Can change features used
y_columns = ['hg/ha_yield']

X = full_df[X_columns]
y = full_df[y_columns]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
def train(model):
	model.fit(X_train, y_train)
	
	return model

initial_model = RandomForestRegressor()
model_and_weights = save_weights(initial_model)
model = model_and_weights[0]
best_hyperparams = model_and_weights[1]
model = train(model)

# Testing and evaluating model

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Prediction functions

def predict_yield(produce, year, area):
	X = [[item_codes[produce], year, area_codes[area]]]
	X = pd.DataFrame(X, columns=['Encoded_item', 'Year', 'Encoded_area'])
	y_pred = model.predict(X)
	
	return y_pred

def optimal_location(produce, year):
	yields = []
	for area in areas:
		predicted_yield = predict_yield(produce, year, area)
		yields.append([predicted_yield, area])
	
	location = max(yields, key=lambda x: x[0])[1]
	confidence = r2
	
	return location, confidence

# User-friendly interface to predict optimal location (with confidence) from produce type and year

produce = input("Enter the type of produce of which you would like to optimise the yield: ")
year = int(input("Enter the year you are considering: "))
print(f"The optimal location for growing {produce} in {year} was {optimal_location(produce, year)[0]}. ")
print(f"The model is {round(optimal_location(produce, year)[1]*100, 2)}% confident of this. ")

# Outputting a .yaml file with information about the model

model_information = best_hyperparams
model_information["Hyperparameter optimisation method"] = "RandomizedSearchCV"
model_information["Model"] = "RandomForestRegressor"

file_path = "model_information.yaml"
with open(file_path, 'w') as file:
	yaml.dump(model_information, file, default_flow_style=False)
