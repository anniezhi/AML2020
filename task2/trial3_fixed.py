## AML Task 2: Disease Classification ##
# Trial 0: outlier, feature selection, scaling, imbalance, classification

# Public Score: 0.72648

import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.covariance import EllipticEnvelope
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
from sklearn.metrics import balanced_accuracy_score
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

if __name__ == '__main__':
	path_X_train = 'data/X_train.csv'
	X_train_org = import_data(path_X_train)
	path_y_train = 'data/y_train.csv'
	y_train_org = import_data(path_y_train).ravel()
	path_X_test = 'data/X_test.csv'
	X_test = import_data(path_X_test)
	#print(X_train.shape)
	
	# Train-Val Split
	X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org, test_size=0.1, random_state=42)

	'''
	# Show X-y correlation
	colors = {0:'red',1:'blue',2:'green'}
	figure = plt.scatter(X_train[:,0],X_train[:,2],c=[colors[y] for y in list(y_train)])
	plt.show()
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

	## Feature selection
	#X_val = X_val[:,feature_selected_variance]
	#X_val = X_val[:,feature_selected_percentile]
	#X_val = X_val[:,feature_selected_kbest]

	## Separate y==0，1，2
	index_c0 = [i for i,y in enumerate(y_train) if y==0]
	index_c1 = [i for i,y in enumerate(y_train) if y==1]
	index_c2 = [i for i,y in enumerate(y_train) if y==2]
	X_train_0 = X_train[index_c0,:]
	X_train_1 = X_train[index_c1,:]
	X_train_2 = X_train[index_c2,:]
	y_train_0 = y_train[index_c0]
	y_train_1 = y_train[index_c1]
	y_train_2 = y_train[index_c2]

	y_hat_val_proba = np.zeros((len(X_val),3,6))
	y_hat_test_proba = np.zeros((len(X_test),3,6))

	for i in range(6):
		#if i!=5:
		#	continue
		#X_train_1_sep, _, y_train_1_sep, _ = train_test_split(X_train_1, y_train_1, test_size=1/6)
		X_train_1_sep = X_train_1[i*int(len(X_train_1)/6):min((i+1)*int(len(X_train_1)/6),len(X_train_1)),:]
		y_train_1_sep = y_train_1[i*int(len(X_train_1)/6):min((i+1)*int(len(X_train_1)/6),len(X_train_1))]
		X_train_comb = np.concatenate((X_train_0, X_train_1_sep, X_train_2),axis=0)
		y_train_comb = np.concatenate((y_train_0, y_train_1_sep, y_train_2),axis=0)
		X_train_comb, y_train_comb = unison_shuffle(X_train_comb, y_train_comb)
		#print(y_train_comb.shape)
		# Classification
		## KNN
		#clf = KNeighborsClassifier(6).fit(X_train, y_train)
		## Linear SVM
		#clf = SVC(kernel='linear',C=0.1).fit(X_train, y_train)
		## RBF SVM
		clf1 = SVC(C=1.6, probability=True).fit(X_train_comb, y_train_comb)
		## Gaussian Process
		#clf = GaussianProcessClassifier(1.0*RBF(1.0)).fit(X_train, y_train)
		## Decision Tree
		#clf = DecisionTreeClassifier(max_depth=12, random_state=42, class_weight='balanced').fit(X_train, y_train)
		## Random Forest
		#clf = RandomForestClassifier(max_depth=12, max_features=None, random_state=42).fit(X_train, y_train)
		## Neural Net
		clf2 = MLPClassifier(hidden_layer_sizes=(600,),random_state=42).fit(X_train_comb, y_train_comb)
		## AdaBoost
		#clf = AdaBoostClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
		## Naive Bayes
		#clf = GaussianNB(var_smoothing=0.006).fit(X_train, y_train)
		## QDA
		#clf = QuadraticDiscriminantAnalysis(reg_param = 0.6).fit(X_train, y_train)
		## XGB
		#clf4 = GradientBoostingClassifier(n_estimators=300).fit(X_train_comb, y_train_comb)
		## LGMB
		clf3 = LGBMClassifier(reg_alpha=1.5).fit(X_train_comb, y_train_comb)
		
		# Predict on train
		#print(clf1.predict_proba(X_train_comb).shape)
		#print(clf2.predict_proba(X_train_comb).shape)
		#y_hat_train_proba = (clf1.predict_proba(X_train_comb) + clf2.predict_proba(X_train_comb) + clf3.predict_proba(X_train_comb) + clf4.predict_proba(X_train_comb))/4
		y_hat_train_proba = (clf1.predict_proba(X_train_comb)+clf2.predict_proba(X_train_comb)+clf3.predict_proba(X_train_comb))/3
		#y_hat_train_proba = clf.predict_proba(X_train_comb)
		y_hat_train = np.argmax(y_hat_train_proba,axis=1)
		score_train = balanced_accuracy_score(y_train_comb, y_hat_train)
		#print(clf1,clf2, clf3, clf4, 'train score', score_train)
		print(clf1, clf2, clf3, 'train score', score_train)
		#print(clf, 'train score', score_train)
		
		# Predict on val
		#y_hat_val_proba[:,:,i] = (clf1.predict_proba(X_val) + clf2.predict_proba(X_val) + clf3.predict_proba(X_val) + clf4.predict_proba(X_val))/3
		y_hat_val_proba[:,:,i] = (clf1.predict_proba(X_val) + clf2.predict_proba(X_val) + clf3.predict_proba(X_val))/3
		#y_hat_val_proba[:,:,i] = clf.predict_proba(X_val)

		# Predict on test
		y_hat_test_proba[:,:,i] = (clf1.predict_proba(X_test) + clf2.predict_proba(X_test) + clf3.predict_proba(X_test))/3

	# Register results
	y_hat_val = np.argmax(np.sum(y_hat_val_proba,axis=2),axis=1)
	#print(y_hat_val)
	#score_train = balanced_accuracy_score(y_train_comb, y_hat_train)
	score_val = balanced_accuracy_score(y_val, y_hat_val)
	#print(clf, 'val score', score_val)
	print(clf1, clf2, clf3, 'val score', score_val)
	#print(clf1, clf2, clf3, clf4, 'val score', score_val)
	
	
	# Test data prediction
	#y_pred_proba = (clf1.predict_proba(X_test) + clf2.predict_proba(X_test) + clf3.predict_proba(X_test)) / 3
	y_pred = np.argmax(np.sum(y_hat_test_proba,axis=2),axis=1)
	np.savetxt('trial3_result.csv',np.dstack((np.arange(y_pred.size),y_pred))[0],"%d,%d",header="id,y")
	