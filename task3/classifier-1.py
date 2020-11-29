## AML Task 3: ECG Classification ##
# remove nan, outlier, feature selection, scaling, imbalance, classification

import numpy as np
from numpy import genfromtxt
from numpy import inf
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectKBest, f_regression, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, TransformerMixin
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from lightgbm import LGBMClassifier

# Import data
def import_data(path):
	data = genfromtxt(path, delimiter=',')
	data = data[1:,1:]
	return data

# Outlier detection
def outlier_detect_lof(X, y, contamination):
	lof = LocalOutlierFactor(n_neighbors=5, contamination=contamination)
	yhat = lof.fit_predict(X)
	mask_outlier = yhat != 1
	mask_nonoutlier = yhat == 1
	return X[mask_nonoutlier,:], y[mask_nonoutlier]

def outlier_detect_iso(X, y, contamination, n_estimators=100):
	iso = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
	yhat = iso.fit_predict(X)
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

def outlier_detect_oneclasssvm(X, y, gamma, nu):
	clf = OneClassSVM(gamma=gamma, nu=nu)
	yhat = clf.fit_predict(X)
	mask_outlier = yhat == -1
	mask_nonoutlier = yhat != -1
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
	X_min = np.expand_dims(np.nanmin(X, axis=0),0)
	X_max = np.expand_dims(np.nanmax(X, axis=1),0)
	X_scaled = 2*(X - X_min) / (X_max - X_min) - 1
	return X
def scaling_standard(X):
	X_mean = np.expand_dims(np.nanmean(X, axis=0),0)
	X_std = np.expand_dims(np.nanstd(X, axis=0),0)
	X_scaled = (X - X_mean) / X_std
	return X_scaled
'''
def scaling_robust(X):
	scaler = RobustScaler()
	X = scaler.fit_transform(X)
	return X
'''

# Oversampling
def oversample_smote(X, y):
	oversample = SMOTE(k_neighbors=11,random_state=42)
	X, y = oversample.fit_resample(X, y)
	return X, y

def oversample_borderline(X, y):
	oversample = BorderlineSMOTE(k_neighbors=11, m_neighbors=25,random_state=42)
	X, y = oversample.fit_resample(X, y)
	return X, y

def oversample_svm(X, y):
	oversample = SVMSMOTE(k_neighbors=9, random_state=42)
	X, y = oversample.fit_resample(X, y)
	return X, y

## Unison Shuffling
def unison_shuffle(a,b):
	assert(len(a)==len(b))
	p = np.random.permutation(len(a))
	return a[p], b[p]

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

if __name__ == '__main__':
	path_X_train = 'data/X_train_features.csv'
	#path_X_train = 'data/X_train_features_mmscaler.csv'
	X_train_org = import_data(path_X_train)
	X_train_org = np.concatenate((X_train_org[:,:55], X_train_org[:,64:]),axis=1)
	
	path_y_train = 'data/y_train.csv'
	y_train_org = import_data(path_y_train).ravel()
	
	path_X_test = 'data/X_test_features.csv'
	#path_X_test = 'data/X_test_features_mmscaler.csv'
	X_test = import_data(path_X_test)
	X_test = np.concatenate((X_test[:,:55], X_test[:,64:]),axis=1)
	#print(X_train_org.shape)

	# Remove inf
	X_train_org[X_train_org == inf] = np.nan
	X_train_org[X_train_org == -inf] = np.nan
	X_test[X_test == inf] = np.nan
	X_test[X_test == -inf] = np.nan

	# Scaling - Train
	#X_train_org = scaling_standard(X_train_org)
	#X_train_org = scaling_minmax(X_train_org)

	# Scaling - Test
	#X_test = scaling_standard(X_test)
	#X_test = scaling_minmax(X_test)

	'''
	# Remove NaN
	X_train_org_nan_list = np.isnan(X_train_org).any(axis=1)
	X_train_org = X_train_org[~X_train_org_nan_list]
	y_train_org = y_train_org[~X_train_org_nan_list]

	y_test = np.zeros(len(X_test))
	X_test_nan_list = np.isnan(X_test).any(axis=1)
	#print(X_test_nan_list)
	y_test[X_test_nan_list] = np.random.choice([0,1,2,3],sum(X_test_nan_list), p=[0.592, 0.087, 0.288, 0.033])
	X_test = X_test[~X_test_nan_list]
	'''

	# Impute NaN
	#X_train_org = impute_simple(X_train_org,'median')
	X_test = impute_simple(X_test,'mean')


	#print(X_train_org.shape)
	
	# Train-Val Split
	X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org, test_size=0.1)
	#print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

	# Impute - Train
	X_train = impute_simple(X_train, 'mean')

	'''
	# Show X-y correlation
	colors = {0:'red',1:'blue',2:'green',3:'orange'}
	figure = plt.scatter(X_train[:,1],X_train[:,8],c=[colors[y] for y in list(y_train)])
	plt.show()
	'''
	'''
	# Outlier Detection
	## LocalOutlierFactor
	#X_train, y_train = outlier_detect_lof(X_train, y_train, float(0.05))
	### Plot outliers
	#plt.scatter(X_train[:,0:1],s=3.)
	#radius = (X_scores.max()-X_scores)/(X_scores.max()-X_scores.min())
	#plt.scatter(X_train[:,0], y_train,s=1000*radius,edgecolors='r',facecolors='none')
	#plt.show()
	
	## IsolationForest
	#X_train, y_train = outlier_detect_iso(X_train, y_train, float(0.05))
	## EllipticEnvelope
	#X_train, y_train = outlier_detect_elliptic(X_train, y_train, float(0.015))
	## OneClassSVM
	#X_train, y_train = outlier_detect_oneclasssvm(X_train, y_train, 'scale', 0.05)

	# Scaling
	#X_train = scaling_minmax(X_train)
	#X_train = scaling_standard(X_train)
	#X_train = scaling_robust(X_train)
	'''

	# Feature Selection
	## VarianceThreshold
	#print(X_train.shape)
	#X_train, feature_selected_variance = feature_sel_variance(X_train, threshold=0.0)
	#print(X_train.shape)
	#print(feature_selected_variance.shape)
	## Percentile
	#X_train, feature_selected_percentile = feature_sel_percentile(X_train, y_train, f_classif, percentile=80)
	#print(X_train.shape)
	#print(feature_selected)
	## KBest
	#X_train, feature_selected_kbest = feature_sel_kbest(X_train, y_train, f_regression, k=600)
	
	# Val data preprocessing
	## Scaling
	#X_val = scaling_minmax(X_val)
	#X_val = scaling_standard(X_val)
	#X_val = scaling_robust(X_val)
	#print(X_val.shape)

	## Impute
	X_val = impute_simple(X_val, 'mean')

	## Feature selection
	#X_val = X_val[:,feature_selected_variance]
	#X_val = X_val[:,feature_selected_percentile]
	#X_val = X_val[:,feature_selected_kbest]

	'''
	## Separate y==0，1，2
	index_c0 = [i for i,y in enumerate(y_train) if y==0]
	index_c1 = [i for i,y in enumerate(y_train) if y==1]
	index_c2 = [i for i,y in enumerate(y_train) if y==2]
	index_c3 = [i for i,y in enumerate(y_train) if y==3]
	#print(len(index_c0), len(index_c1), len(index_c2), len(index_c3))
	X_train_0 = X_train[index_c0,:]
	X_train_1 = X_train[index_c1,:]
	X_train_2 = X_train[index_c2,:]
	X_train_3 = X_train[index_c3,:]
	#print(X_train_0.shape, X_train_1.shape, X_train_2.shape, X_train_3.shape)
	y_train_0 = y_train[index_c0]
	y_train_1 = y_train[index_c1]
	y_train_2 = y_train[index_c2]
	y_train_3 = y_train[index_c3]
	#print(len(y_train_0), len(y_train_1), len(y_train_2), len(y_train_3))

	## Separate bootstrap size
	n_sample = 20
	'''
	#y_hat_val_proba = np.zeros((len(X_val),4,n_sample))
	#y_hat_test_proba = np.zeros((len(X_test),4,n_sample))
	#score_train_all = []
	

	# Scaling - Train
	#X_train_comb = scaling_standard(X_train_comb)

	# Impute - Train
	#X_train_comb = impute_simple(X_train_comb,'mean')

	# Classification
	## KNN
	#clf = KNeighborsClassifier(6).fit(X_train, y_train)
	## Linear SVM
	#clf = SVC(kernel='linear',C=0.1).fit(X_train, y_train)
	## RBF SVM
	#clf1 = SVC(C=1.6, probability=True).fit(X_train_comb, y_train_comb)
	## Gaussian Process
	#clf = GaussianProcessClassifier(1.0*RBF(1.0)).fit(X_train, y_train)
	## Decision Tree
	#clf = DecisionTreeClassifier(max_depth=12, random_state=42, class_weight='balanced').fit(X_train, y_train)
	## Random Forest
	#clf = RandomForestClassifier(max_depth=12, max_features=None, random_state=42).fit(X_train, y_train)
	## Neural Net
	#clf2 = MLPClassifier(hidden_layer_sizes=(600,),random_state=42).fit(X_train_comb, y_train_comb)
	## AdaBoost
	#clf = AdaBoostClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
	## Naive Bayes
	#clf = GaussianNB(var_smoothing=0.006).fit(X_train, y_train)
	## QDA
	#clf = QuadraticDiscriminantAnalysis(reg_param = 0.6).fit(X_train, y_train)
	## XGB
	clf = GradientBoostingClassifier(n_estimators=70).fit(X_train, y_train)
	## LGMB
	#clf3 = LGBMClassifier(reg_alpha=1.5).fit(X_train_comb, y_train_comb)
		
	# Predict on train
	#print(clf1.predict_proba(X_train_comb).shape)
	#print(clf2.predict_proba(X_train_comb).shape)
	#y_hat_train_proba = (clf1.predict_proba(X_train_comb) + clf2.predict_proba(X_train_comb) + clf3.predict_proba(X_train_comb) + clf4.predict_proba(X_train_comb))/4
	#y_hat_train_proba = (clf1.predict_proba(X_train_comb)+clf2.predict_proba(X_train_comb)+clf3.predict_proba(X_train_comb))/3
	y_hat_train_proba = clf.predict_proba(X_train)
	y_hat_train = np.argmax(y_hat_train_proba,axis=1)
	score_train = f1_score(y_train, y_hat_train, average='micro')
	#print(clf1,clf2, clf3, clf4, 'train score', score_train)
	#print(clf1, clf2, clf3, 'train score', score_train)
	print('train score', score_train)
	
	# Predict on val
	#y_hat_val_proba[:,:,i] = (clf1.predict_proba(X_val) + clf2.predict_proba(X_val) + clf3.predict_proba(X_val) + clf4.predict_proba(X_val))/3
	#y_hat_val_proba[:,:,i] = (clf1.predict_proba(X_val) + clf2.predict_proba(X_val) + clf3.predict_proba(X_val))/3
	y_hat_val_proba = clf.predict_proba(X_val)
	
	# Predict on test
	#y_hat_test_proba[:,:,i] = (clf1.predict_proba(X_test) + clf2.predict_proba(X_test) + clf3.predict_proba(X_test))/3
	#y_hat_test_proba[:,:,i] = clf.predict_proba(X_test)

	
	# Register results - Val
	## boosting - average
	y_hat_val = np.argmax(y_hat_val_proba,axis=1)
	## boosting - voting
	#y_hat_val = np.argmax(np.max(y_hat_val_proba, axis=2), axis=1)
	#print(y_hat_val)
	#score_train = balanced_accuracy_score(y_train_comb, y_hat_train)
	score_val = f1_score(y_val, y_hat_val, average='micro')
	print(clf, 'val score', score_val)
	#print(clf1, clf2, clf3, 'val score', score_val)
	#print(clf1, clf2, clf3, clf4, 'val score', score_val)
	
	
	# Test data prediction
	#y_pred_proba = (clf1.predict_proba(X_test) + clf2.predict_proba(X_test) + clf3.predict_proba(X_test)) / 3
	y_pred_proba = clf.predict_proba(X_test)
	y_pred = np.argmax(y_pred_proba,axis=1)
	#y_test[~X_test_nan_list] = y_pred
	#np.savetxt('result_3.csv',np.dstack((np.arange(y_test.size),y_test))[0],"%d,%d",header="id,y")
	np.savetxt('result_4.csv',np.dstack((np.arange(y_pred.size),y_pred))[0],"%d,%d",header="id,y")
	