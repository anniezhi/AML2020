## AML Task 1: Brain Age Detection ##
# Trial 1: impute, outlier, feature selection, scaling, regression

import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, BayesianRidge, TweedieRegressor
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, TransformerMixin
from xgboost import XGBRegressor

# Import data
def import_data(path):
	data = genfromtxt(path, delimiter=',')
	data = data[1:,1:]
	return data

# Missing value imputation
def impute_knn(X, n_neighbors, weights):
	imputer_knn = KNNImputer(n_neighbors=n_neighbors, weights=weights)
	X = imputer_knn.fit_transform(X)
	return X
def impute_simple(X, strategy):
	imputer_simple = SimpleImputer(missing_values=np.nan, strategy=strategy)
	X = imputer_simple.fit_transform(X)
	return X
def impute_iterative(X):
	imputer_iterative = IterativeImputer(random_state=0)
	X = imputer_iterative.fit_transform(X)
	return X

# Outlier detection
class OutlierExtractor(TransformerMixin):
	def __init__(self, **kwargs):
		self.threshold = kwargs.pop('neg_conf_val',-10.0)
		self.kwargs = kwargs
		self.X = None
	def transform(self, X, y):
		X = np.asarray(X)
		y = np.asarray(y)
		lcf = LocalOutlierFactor(**self.kwargs)
		lcf.fit(X)
		return(X[lcf.negative_outlier_factor_>self.threshold,:],
			   y[lcf.negative_outlier_factor_>self.threshold])
	def fit(self, *args, **kwargs):
		return self

def outlier_detect_iso(X, y, contamination):
	iso = IsolationForest(contamination=contamination, random_state=42)
	yhat = iso.fit_predict(X_train)
	#print(iso.score_samples(X_train).shape)
	mask_outlier = yhat != 1
	mask_nonoutlier = yhat == 1
	'''
	### Plot outliers
	plt.scatter(X_train[mask_outlier,20],y_train[mask_outlier])
	plt.scatter(X_train[mask_nonoutlier,20],y_train[mask_nonoutlier],color='r')
	plt.show()
	'''
	return X[mask_nonoutlier,:],y[mask_nonoutlier]

def outlier_detect_elliptic(X, y, contamination):
	cov = EllipticEnvelope(contamination=contamination, random_state=0)
	yhat = cov.fit_predict(X)
	mask_outlier = yhat != 1
	mask_nonoutlier = yhat == 1
	return X[mask_nonoutlier,:], y[mask_nonoutlier]

# Featur selection
def feature_sel_variance(X_train, threshold):
	sel = VarianceThreshold(threshold=threshold)
	X_train = sel.fit_transform(X_train)
	feature_selected = sel.get_support(indices=True)
	return X_train, feature_selected

def feature_sel_percentile(X_train, y, score_func, percentile):
	sel = SelectPercentile(score_func=score_func, percentile=percentile)
	X_train = sel.fit_transform(X_train, np.ravel(y))
	feature_selected = sel.get_support(indices=True)
	return X_train, feature_selected

def feature_sel_kbest(X_train, y, score_func, k):
	sel = SelectKBest(score_func=score_func, k=k)
	X_train = sel.fit_transform(X_train, np.ravel(y))
	feature_selected = sel.get_support(indices=True)
	return X_train, feature_selected

# Scaling
def scaling_minmax(X):
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X)
	return X
def scaling_standard(X):
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	return X
def scaling_robust(X):
	scaler = RobustScaler()
	X = scaler.fit_transform(X)
	return X



if __name__ == '__main__':
	path_X_train = 'data/X_train.csv'
	X_train = import_data(path_X_train)
	path_y_train = 'data/y_train.csv'
	y_train = import_data(path_y_train)
	path_X_test = 'data/X_test.csv'
	X_test = import_data(path_X_test)
	#print(X_train.shape)
	
	# Train-Val Split
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


	# Missing Data Impute
	## KNN
	#X_train = impute_knn(X_train, 15, 'uniform')
	## Simple imputer
	X_train = impute_simple(X_train, 'median')
	## Iterative imputer
	#X_train = impute_iterative(X_train)
	'''
	# Show X-y correlation
	figure = plt.scatter(X_train[:,200],y_train)
	plt.show()
	'''
	
	# Outlier Detection
	'''
	## LocalOutlierFactor
	clf = LocalOutlierFactor(n_neighbors=5)
	clf.fit_predict(np.concatenate((y_train,np.expand_dims(X_train,axis=1))))
	X_scores = clf.negative_outlier_factor_
	### Plot outliers
	plt.scatter(X_train[:,0],y_train, s=3.)
	radius = (X_scores.max()-X_scores)/(X_scores.max()-X_scores.min())
	plt.scatter(X_train[:,0], y_train,s=1000*radius,edgecolors='r',facecolors='none')
	plt.show()
	'''

	## IsolationForest
	#X_train, y_train = outlier_detect_iso(X_train, y_train, float(0.2))
	#print(X_train.shape)
	#print(y_train.shape)

	## EllipticEnvelope
	#X_train, y_train = outlier_detect_elliptic(X_train, y_train, float(0.005))

	# Feature Selection
	## VarianceThreshold
	#print(X_train.shape)
	#X_train, feature_selected_variance = feature_sel_variance(X_train, threshold=0.0)
	#print(X_train.shape)
	#print(feature_selected_variance.shape)
	## Percentile
	#X_train, feature_selected_percentile = feature_sel_percentile(X_train, y_train, f_regression, percentile=70)
	#print(X_train.shape)
	#print(feature_selected)
	## KBest
	X_train, feature_selected_kbest = feature_sel_kbest(X_train, y_train, f_regression, k=130)


	# Scaling
	#X_train = scaling_minmax(X_train)
	#X_train = scaling_standard(X_train)
	X_train = scaling_robust(X_train)

	# Regression
	## Linear
	#reg = LinearRegression().fit(X_train, y_train)
	## Ridge
	#reg = Ridge(alpha=2).fit(X_train,y_train)
	## Lasso
	#reg = Lasso(alpha=.1).fit(X_train, y_train)
	## ElasticNet
	#reg = ElasticNet(random_state=42).fit(X_train, y_train)
	## LassoLars
	#reg = LassoLars(alpha=0.05).fit(X_train, y_train)
	## BayesianRidge
	#reg = BayesianRidge().fit(X_train, y_train)
	## TweedieRegressor
	#reg = TweedieRegressor(power=2,alpha=0.5,link='log',max_iter=300).fit(X_train, y_train)
	## RANSAC
	#reg = RANSACRegressor(random_state=42).fit(X_train, y_train)
	## Theil-Sen
	#reg = TheilSenRegressor(random_state=42).fit(X_train, y_train)
	## Huber
	#reg = HuberRegressor().fit(X_train, y_train)
	## GradientBoostingRegressor
	#reg = GradientBoostingRegressor(random_state=42, n_estimators=300).fit(X_train, y_train)
	## XGBRegressor
	#reg = XGBRegressor(random_state=42, learning_rate=0.05, base_score=0.1, max_depth=10, n_estimators=250).fit(X_train, y_train)
	## SVR
	reg = SVR(C=50.0).fit(X_train, y_train)
	#y_hat_train = reg.predict(X_train)
	#print(r2_score(X_train, y_train))
	#print(X_train.shape[1]*X_train.var())

	# Val data preprocessing
	## Impute
	#X_val = impute_knn(X_val,15, 'uniform')
	X_val = impute_simple(X_val, 'median')
	#X_val = impute_iterative(X_val)
	## Feature selection
	#X_val = X_val[:,feature_selected_variance]
	#X_val = X_val[:,feature_selected_percentile]
	X_val = X_val[:,feature_selected_kbest]
	## Scaling
	#X_val = scaling_minmax(X_val)
	#X_val = scaling_standard(X_val)
	X_val = scaling_robust(X_val)
	#print(X_val.shape)
	# Predict
	y_hat = reg.predict(X_val)

	# Print results
	print(reg, 'train score',reg.score(X_train, y_train), 'val score', r2_score(y_val, y_hat))
	#print(reg, 'val score', r2_score(y_val, y_hat))

	
	# Test data prediction
	X_test = impute_simple(X_test, 'median')
	#X_test = impute_knn(X_test,15,'uniform')
	X_test = X_test[:,feature_selected_kbest]
	X_test = scaling_robust(X_test)
	y_pred = reg.predict(X_test)
	np.savetxt('trial1_result.csv',np.dstack((np.arange(y_pred.size),y_pred))[0],"%d,%f",header="id,y")
	