from joblib import dump

def save_weights():
	model_information = best_hyperparams
	model_information["Hyperparameter optimisation method"] = "RandomizedSearchCV"
	model_information["Model"] = "RandomForestRegressor"
	
	file_path = "model_information.yaml"
	with open(file_path, 'w') as file:
		yaml.dump(model_information, file, default_flow_style=False)
	
	dump(model, 'model.joblib')