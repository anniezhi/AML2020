import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
# impute data
from sklearn.impute import SimpleImputer
# scaling the data
from sklearn.preprocessing import RobustScaler, StandardScaler
# different feature selection techniques
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectKBest, RFECV, f_regression 
# different outlier detection techniques
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
# regression classifiers
from sklearn.svm import LinearSVR, SVR

#### IMPORT DATA ####
# import data
df_train = pd.read_csv('X_train.csv')
df_data_to_predict = pd.read_csv('X_test.csv')
df_y_train = pd.read_csv('y_train.csv')

# convert to np arrays
X = df_train.iloc[:,1:].to_numpy()
y = df_y_train.iloc[:,1:].to_numpy().reshape(-1)
X_to_predict = df_data_to_predict.iloc[:,1:].to_numpy()

# impute missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X,y)
X = imputer.transform(X)
X_to_predict = imputer.transform(X_to_predict)

#### OUTLIER DETECTION ####
## Local Outlier Factor ##
#outliers = LocalOutlierFactor().fit_predict(X)
## Isolation forest ##
outliers = IsolationForest(random_state=0).fit_predict(X)
# only keep inliners in training set for regression classifier
X = X[np.where(outliers == 1)[0]].copy()
y = y[np.where(outliers == 1)[0]].copy()

#### FEATURE SELECTION ####

## Variance Threshold ##
# find features which have 0 variance in training data
selector = VarianceThreshold()
selector.fit(X)
# save column ids of feature vectors
features_to_delete = np.where(selector.variances_ == 0)[0]
# delete features from X_train and X_test
X = np.delete(X, features_to_delete, axis=1)
#X_test = np.delete(X_test, features_to_delete, axis=1)
X_to_predict = np.delete(X_to_predict, features_to_delete, axis=1)

#### PIPELINE ####
# make a pipeline of the process
# include standard scaling, feature extraction, missining value imputation
highest_mean = -100
#Cs = np.array([0.0001,0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000])
Cs = np.arange(51,52,0.1)
#gammas = [0.023,0.024,0.025]
#for i, c in enumerate(Cs):
regr = make_pipeline(RobustScaler(), SelectKBest(f_regression, k=73), SVR(C=51, gamma=0.023))
    #regr = make_pipeline(StandardScaler(), RFECV(SVR(kernel='linear'), verbose=1), SVR())
    #### TRAIN THE MODEL ####
    # run cross validation
cv_results = cross_validate(regr, X, y, scoring=('r2'))
if cv_results['test_score'].mean() > highest_mean:
    highest_mean = cv_results['test_score'].mean()
    #print('Number of features: {},  Mean: {}'.format(i, highest_mean))
    print(highest_mean)
    #print('C: {},  Mean: {}'.format(c, highest_mean))
    #print('C: {}, Mean: {}'.format(i, cv_results['test_score'].mean()))
#regr = regr.fit(X_train, y_train)

# predict new data with best found predictor
regr = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=73), SVR(C=51, gamma=0.023))
regr = regr.fit(X ,y)
y_pred = regr.predict(X_to_predict)

#### FINAL FILE ####
# convert the predictions into the desired file format with ids
df_ids = pd.DataFrame(df_data_to_predict['id'])
df_predictions = df_ids.join(pd.DataFrame(y_pred, columns=['y']))

# save the predictions dataframe
df_predictions.to_csv('predictions.csv', index=False)