import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample
from sklearn.utils import shuffle
# Scaler
from sklearn.preprocessing import StandardScaler
# feature selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
# classifiers
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

#### IMPORT DATA ####
# import data
df_train = pd.read_csv('X_train.csv')
df_data_to_predict = pd.read_csv('X_test.csv')
df_y_train = pd.read_csv('y_train.csv')

# convert to np arrays
X = df_train.iloc[:,1:].to_numpy()
y = df_y_train.iloc[:,1:].to_numpy().reshape(-1)
X_to_predict = df_data_to_predict.iloc[:,1:].to_numpy()


##### UP / DOWN SAMPLING ####
#
## Up sampling
#amount_of_samples = X[y==1].shape[0]
#
## split X according to its classes
#X_feature_0 = X[y==0]
#X_feature_1 = X[y==1]
#X_feature_2 = X[y==2]
#
## upsample the smaller classes 0 and 2
#X_feature_0 = resample(X_feature_0, n_samples=amount_of_samples, random_state=42)
#X_feature_2 = resample(X_feature_2, n_samples=amount_of_samples, random_state=42)
#
## create ys
#y_0 = np.zeros(amount_of_samples).astype(int)
#y_1 = np.ones(amount_of_samples).astype(int)
#y_2 = np.empty(amount_of_samples)
#y_2.fill(2)
#y_2 = y_2.astype(int)
#
## append all values
#X = np.concatenate((X_feature_0, X_feature_1, X_feature_2), axis=0)
#y = np.concatenate((y_0, y_1, y_2), axis=0)
#
## shuffle data
#X, y = shuffle(X, y, random_state=42)

## Down sampling
#amount_of_samples = X[y==0].shape[0]
#
#X_feature_0 = X[y==0]
#X_feature_1 = X[y==1]
#X_feature_2 = X[y==2]
#
#X_feature_1 = resample(X_feature_1, n_samples=amount_of_samples, random_state=42)
#
#y_0 = np.zeros(amount_of_samples).astype(int)
#y_1 = np.ones(amount_of_samples).astype(int)
#y_2 = np.empty(amount_of_samples)
#y_2.fill(2)
#y_2 = y_2.astype(int)
#
#X = np.concatenate((X_feature_0, X_feature_1, X_feature_2), axis=0)
#y = np.concatenate((y_0, y_1, y_2), axis=0)
#
#X, y = shuffle(X, y, random_state=42)


### CLASSIFICATION ###
# SVC classifier
params = {"svc__C": [0.1, 1, 10], "svc__gamma": [0.0001, 0.001]}
regr = make_pipeline(StandardScaler(), SVC(class_weight='balanced'))#OneVsRestClassifier(SVC(C=1, class_weight='balanced')))
cv_results = cross_validate(regr, X, y, scoring=('balanced_accuracy'), verbose=10)
#grid_search = GridSearchCV(regr, params, scoring='balanced_accuracy', verbose=10)
#grid_search.fit(X,y)
mean = cv_results['test_score'].mean()
#print('gamma: {},  Mean: {}'.format(g, mean))
#if mean > highest_mean:
    #best_g = g
#    highest_mean = mean
#print('Best gamma: {}, with mean: {}'.format(best_g, highest_mean))
print(mean)


## Random Forest Classifier
#regr = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, class_weight='balanced'))
#cv_results = cross_validate(regr, X, y, scoring=('balanced_accuracy'), verbose=10)
#highest_mean = cv_results['test_score'].mean()
#print(highest_mean)


### PREDICTION ###
# predict new data with best found predictor
regr = make_pipeline(StandardScaler(), SVC(C=1))#,class_weight='balanced'))
regr = regr.fit(X ,y)
y_pred = regr.predict(X_to_predict)


#### FINAL FILE ####
# convert the predictions into the desired file format with ids
df_ids = pd.DataFrame(df_data_to_predict['id'])
df_predictions = df_ids.join(pd.DataFrame(y_pred, columns=['y']))


# save the predictions dataframe
df_predictions.to_csv('predictions.csv', index=False)