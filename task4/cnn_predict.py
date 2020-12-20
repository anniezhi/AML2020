## AML Task 4: Sleep Stage Classification ##
# CNN: Predict for test case

import numpy as np
from numpy import genfromtxt
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.metrics import balanced_accuracy_score
import h5py

if __name__ == '__main__':
	# import data - test
	path_eeg1_test_sub1 = 'data_preprocessed/epochs_eeg1_test_sub1.npz'
	path_eeg2_test_sub1 = 'data_preprocessed/epochs_eeg2_test_sub1.npz'
	path_emg_test_sub1 = 'data_preprocessed/epochs_emg_test_sub1.npz'
	eeg1_test_sub1 = np.load(path_eeg1_test_sub1)['arr_0']
	eeg2_test_sub1 = np.load(path_eeg2_test_sub1)['arr_0']
	emg_test_sub1 = np.load(path_emg_test_sub1)['arr_0']

	path_eeg1_test_sub2 = 'data_preprocessed/epochs_eeg1_test_sub2.npz'
	path_eeg2_test_sub2 = 'data_preprocessed/epochs_eeg2_test_sub2.npz'
	path_emg_test_sub2 = 'data_preprocessed/epochs_emg_test_sub2.npz'
	eeg1_test_sub2 = np.load(path_eeg1_test_sub2)['arr_0']
	eeg2_test_sub2 = np.load(path_eeg2_test_sub2)['arr_0']
	emg_test_sub2 = np.load(path_emg_test_sub2)['arr_0']


	# import models
	model_12 = keras.models.load_model("model_12.h5")
	model_23 = keras.models.load_model("model_23.h5")
	model_13 = keras.models.load_model("model_13.h5")


	# reshape & expand dim
	for sig in ['eeg1', 'eeg2', 'emg']:
		for sub in ['1','2']:
			vars()[sig+'_test_sub'+sub] = vars()[sig+'_test_sub'+sub].reshape(-1, 160, 48, 1)

	# combine train data to 3 channels
	for sub in ['1','2']:
		vars()['input_test_sub'+sub] = np.concatenate((vars()['eeg1_test_sub'+sub],vars()['eeg2_test_sub'+sub],vars()['emg_test_sub'+sub]),
			axis=3)

	# combine train data of subs
	input_test = np.concatenate((input_test_sub1, input_test_sub2),axis=0)

	# predict
	pred_model_12 = model_12.predict(input_test, verbose=0)
	pred_model_23 = model_23.predict(input_test, verbose=0)
	pred_model_13 = model_13.predict(input_test, verbose=0)

	# label predict
	label_pred_model_12 = np.argmax(np.round(pred_model_12), axis=1) + 1
	label_pred_model_23 = np.argmax(np.round(pred_model_23), axis=1) + 1
	label_pred_model_13 = np.argmax(np.round(pred_model_13), axis=1) + 1

	label_pred_model_all = np.argmax(np.round(pred_model_12+pred_model_23+pred_model_13),axis=1) + 1

	# save predictions
	np.save('label_pred_test_model_12.npy',label_pred_model_12)
	np.save('label_pred_test_model_23.npy',label_pred_model_23)
	np.save('label_pred_test_model_13.npy',label_pred_model_13)

	np.save('label_pred_test_model_all.npy',label_pred_model_all)
