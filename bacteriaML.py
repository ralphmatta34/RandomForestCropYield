import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Accessing and cleaning up our data
metadata = pd.read_table('sample_metadata.tsv')
metadata.index = ['farm_%i' %i for i in range(len(metadata))]

sequences_counts = pd.read_table('16S_counts.tsv')
sequences_counts.index = ['farm_%i' % i for i in range(len(sequences_counts))]

bacteria_counts = pd.read_table('bacteria_counts.tsv')
bacteria_counts.index = ['farm_%i' % i for i in range(len(bacteria_counts))]

cols = list(bacteria_counts.columns)
np.random.seed(42)
np.random.shuffle(cols)
bacteria_counts = bacteria_counts[cols]
sequence_to_species_dict = np.load('sequence_to_species_dict.npy', allow_pickle=True).item()
bacteria_counts = bacteria_counts.drop(['Unnamed: 0'], axis=1)

# ax = sns.histplot(data=metadata['crop_yield'])
# ax.set(xlabel='Crop yield / kg/ha', ylabel='# farms')
# plt.show()

# Removing low-prevalence bacteria and log normalising
low_prev_bacteria = []
bacteria = bacteria_counts.columns

for bacterium in bacteria:
  if sum(bacteria_counts[bacterium] > 0) < 10:
    low_prev_bacteria += [bacterium]

bacteria_counts_no_low_prev = bacteria_counts.drop(low_prev_bacteria, axis=1)
bacteria_counts_lognorm = np.log(bacteria_counts_no_low_prev + 1)
bacteria_counts_lognorm.to_csv('bacteria_counts_lognorm.csv')

sequences_counts_t = sequences_counts.transpose()
sequences_counts_t['species'] = [sequence_to_species_dict[i] for i in sequences_counts_t.index]
summed_data = sequences_counts_t.groupby('species').sum()
bacteria_counts = summed_data.transpose()

# Preparing our data for the model
X = bacteria_counts_lognorm
y = metadata['crop_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Decision Tree
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(y_test, preds, '.')
plt.xlabel('True crop yields / kg/ha')
plt.ylabel('Predicted crop yields / kg/ha')
xs = [0.0, 1.2]
ys = [0.0, 1.2]
plt.plot(xs, ys, label='y=x')
m, c = np.polyfit(y_test, preds, 1)
plt.plot(y_test, m*y_test+c, label='line of best fit')
plt.legend(loc='upper left')
plt.title('Decision Tree')
#plt.show()
decision_tree_R2 = r2_score(y_test, preds)
print(f'Decision tree R2 = {round(decision_tree_R2, 3)}. ')

# plt.figure(figsize=(20, 20))
# plot_tree(model, max_depth=3, feature_names=list(X.columns))
# plt.show()

# feature_importance_dataframe = pd.DataFrame(model.feature_importances_, columns=['feature_importance'])
# feature_importance_dataframe.index = X.columns
# feature_importance_dataframe_sorted = feature_importance_dataframe.sort_values('feature_importance', ascending=False)
# print(feature_importance_dataframe_sorted.head())

# MLP
model = MLPRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(y_test, preds, '.')
plt.xlabel('True crop yields / kg/ha')
plt.ylabel('Predicted crop yields / kg/ha')
xs = [0.0, 1.2]
ys = [0.0, 1.2]
plt.plot(xs, ys, label='y=x')
m, c = np.polyfit(y_test, preds, 1)
plt.plot(y_test, m*y_test+c, label='line of best fit')
plt.legend(loc='upper left')
plt.title("MLP")
#plt.show()
MLP_R2 = r2_score(y_test, preds)
print(f'MLP R2 = {round(MLP_R2, 3)}. ')

# KNN
model = KNeighborsRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(y_test, preds, '.')
plt.xlabel('True crop yields / kg/ha')
plt.ylabel('Predicted crop yields / kg/ha')
xs = [0.0, 1.2]
ys = [0.0, 1.2]
plt.plot(xs, ys, label='y=x')
m, c = np.polyfit(y_test, preds, 1)
plt.plot(y_test, m*y_test+c, label='line of best fit')
plt.legend(loc='upper left')
plt.title("KNN")
#plt.show()
KNN_R2 = r2_score(y_test, preds)
print(f'KNN R2 = {round(KNN_R2, 3)}. ')

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(y_test, preds, '.')
plt.xlabel('True crop yields / kg/ha')
plt.ylabel('Predicted crop yields / kg/ha')
xs = [0.0, 1.2]
ys = [0.0, 1.2]
plt.plot(xs, ys, label='y=x')
m, c = np.polyfit(y_test, preds, 1)
plt.plot(y_test, m*y_test+c, label='line of best fit')
plt.legend(loc='upper left')
plt.title("Linear Regression")
#plt.show()
linear_regression_R2 = r2_score(y_test, preds)
print(f'Linear regression R2 = {round(linear_regression_R2, 3)}. ')

# Random Forest Regression
model = RandomForestRegressor(n_estimators=10)
model.fit(X_train, y_train)
preds = model.predict(X_test)
plt.plot(y_test, preds, '.')
plt.xlabel('True crop yields / kg/ha')
plt.ylabel('Predicted crop yields / kg/ha')
xs = [0.0, 1.2]
ys = [0.0, 1.2]
plt.plot(xs, ys, label='y=x')
m, c = np.polyfit(y_test, preds, 1)
plt.plot(y_test, m*y_test+c, label='line of best fit')
plt.legend(loc='upper left')
plt.title("Random Forest Regression")
#plt.show()
random_forest_R2 = r2_score(y_test, preds)
print(f'Random forest regression R2 = {round(random_forest_R2, 3)}. ')

# Formatting and outputting R2 scores for each model
all_R2_scores_dict = {decision_tree_R2: 'Decision tree',
                      MLP_R2: 'MLP',
                      KNN_R2: 'KNN',
                      linear_regression_R2: 'Linear regression',
                      random_forest_R2: 'Random forest regression'}
sorted_R2_scores_list = sorted(all_R2_scores_dict.items())
sorted_R2_scores_dict = dict(sorted_R2_scores_list)
final_R2_scores_list = list(enumerate(list(sorted_R2_scores_dict.items())))

# Ranking models
print("In order of best performance (based on R2), the following are the models we used: ")
for index, (R2_score, model_type) in final_R2_scores_list[::-1]:
  print(f'{len(all_R2_scores_dict) - index}. {model_type}')

print(f"\nThe model that performed the best (based on R2) was the {final_R2_scores_list[-1][1][1].lower()}. ")

# Outputting feature importance information
feature_importances = model.feature_importances_
num_possible_features = len(feature_importances)
num_features_used = sum(feature_importances != 0)
print(f"The random forest model used {num_features_used} out of a possible {num_possible_features} features. ")

feature_importance_dataframe = pd.DataFrame(model.feature_importances_, columns=['feature_importance'])
feature_importance_dataframe.index = X.columns
feature_importance_dataframe_sorted = feature_importance_dataframe.sort_values('feature_importance', ascending=False)
nonzero_feature_importance_dataframe_sorted = feature_importance_dataframe_sorted.loc[(feature_importance_dataframe_sorted!=0).any(axis=1)]
print(nonzero_feature_importance_dataframe_sorted.head(15))

y_pred_train = model.predict(X_train)
training_R2 = r2_score(y_train, y_pred_train)

y_pred_test = model.predict(X_test)
testing_R2 = r2_score(y_test, y_pred_test)

print(f"Training R2: {round(training_R2, 3)}. ")
print(f"Testing R2: {round(testing_R2, 3)}. ")

if training_R2 > testing_R2:
  print("Overfitting has taken place. ")

# Pruning for feature reduction via K-fold cross validation

print("Pruning taking place...\n" +
      "K-fold cross validation taking place...\n" +
      "Cross-complexity pruning alpha value being optimised...")

model = RandomForestRegressor(n_estimators=10)
multiplied_ccp_alpha_range = list(range(1, 10, 1))
ccp_alpha_range = [value/(10**5) for value in multiplied_ccp_alpha_range]
model_cv = GridSearchCV(model, param_grid={'ccp_alpha': ccp_alpha_range})
model_cv.fit(X_train, y_train)

KFCV_y_pred_train = model_cv.predict(X_train)
KFCV_training_R2_score = r2_score(y_train, KFCV_y_pred_train)
KFCV_y_pred_test = model_cv.predict(X_test)
KFCV_testing_R2_score = r2_score(y_test, KFCV_y_pred_test)

num_non_zero_features = sum(model_cv.best_estimator_.feature_importances_ != 0)

print(f"Best value of ccp_alpha = {model_cv.best_estimator_.ccp_alpha}. ")
print(f"K-fold cross validated pruned model training R2 =  {round(KFCV_training_R2_score, 3)}. ")
print(f"K-fold cross validation pruned model testing R2 = {round(KFCV_testing_R2_score, 3)}. ")

if KFCV_training_R2_score > KFCV_testing_R2_score:
  print("Overfitting has taken place. ")

print(f"Number of non-zero feature importances in best (well-pruned) model = {num_non_zero_features}. ")

# Finding the 10 farms with the most crop yield
crop_yield_test_predictions = model_cv.predict(X_test)
yield_predictions_dataframe = pd.DataFrame(data={'farm':X_test.index, 'crop_yield':crop_yield_test_predictions})
sorted_dataframe = yield_predictions_dataframe.sort_values('crop_yield', ascending=False)
top_farms_dataframe = sorted_dataframe.head(10)
farm_names = top_farms_dataframe['farm']
print("According to the model, the top 10 best plots are: ")
for index, farm_name in enumerate(farm_names.to_list()):
  print(f"\t{index+1}. Farm {farm_name[5:]}")

# Comparison with lottery crop yield and attendant conclusion
crop_yields_top_10_predictions = top_farms_dataframe['crop_yield']
total_crop_yield_using_model = sum(crop_yields_top_10_predictions)
print(f"\nTotal crop yield using model suggestions: {round(total_crop_yield_using_model, 3)} kg/ha.")

np.random.seed(random.randint(1, 50))
crop_yields_10_random_picks = np.random.choice(y_test, 10)
total_crop_yield_using_lottery = sum(crop_yields_10_random_picks)
print(f"Total crop yield using random picks: {round(total_crop_yield_using_lottery, 3)} kg/ha. ")

if total_crop_yield_using_lottery <= total_crop_yield_using_model:
  percentage_difference = total_crop_yield_using_model/total_crop_yield_using_lottery - 1
  print(f"The model leads to a {round(percentage_difference * 100, 3)}% higher yield than the lottery. \n" +
        f"The hyperparameters are strong and the model should be deployed. ")
else:
  percentage_difference = total_crop_yield_using_lottery/total_crop_yield_using_model - 1
  print(f"The model leads to a {round(percentage_difference*100, 3)}% lower yield than the lottery. ")
