## AML Task 4: Sleep Stage Classification ##
# CNN: Fourier transform, PSD, bandpass, (energy), log, per-freq standardization, epoch split, neighbor concat

import numpy as np
import pandas as pd
from numpy import genfromtxt
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import Softmax
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import balanced_accuracy_score
import h5py

def unison_shuffle(a,b):
	assert(len(a)==len(b))
	p = np.random.permutation(len(a))
	return a[p], b[p]


if __name__ == '__main__':
	# import data - train sub1 & sub2 (train)
	path_eeg1_train_sub1 = 'data_preprocessed/epochs_eeg1_train_sub1.npz'
	path_eeg2_train_sub1 = 'data_preprocessed/epochs_eeg2_train_sub1.npz'
	path_emg_train_sub1 = 'data_preprocessed/epochs_emg_train_sub1.npz'
	eeg1_train_sub1 = np.load(path_eeg1_train_sub1)['arr_0']
	eeg2_train_sub1 = np.load(path_eeg2_train_sub1)['arr_0']
	emg_train_sub1 = np.load(path_emg_train_sub1)['arr_0']

	path_label_train_sub1 = 'data_preprocessed/labels_train_sub1.npz'
	label_train_sub1 = np.load(path_label_train_sub1)['arr_0'] - 1     # to match with to_categorical()

	path_eeg1_train_sub2 = 'data_preprocessed/epochs_eeg1_train_sub2.npz'
	path_eeg2_train_sub2 = 'data_preprocessed/epochs_eeg2_train_sub2.npz'
	path_emg_train_sub2 = 'data_preprocessed/epochs_emg_train_sub2.npz'
	eeg1_train_sub2 = np.load(path_eeg1_train_sub2)['arr_0']
	eeg2_train_sub2 = np.load(path_eeg2_train_sub2)['arr_0']
	emg_train_sub2 = np.load(path_emg_train_sub2)['arr_0']

	path_label_train_sub2 = 'data_preprocessed/labels_train_sub2.npz'
	label_train_sub2 = np.load(path_label_train_sub2)['arr_0'] - 1		# to match with to_categorical()

	# import data - train sub3 (val)
	path_eeg1_train_sub3 = 'data_preprocessed/epochs_eeg1_train_sub3.npz'
	path_eeg2_train_sub3 = 'data_preprocessed/epochs_eeg2_train_sub3.npz'
	path_emg_train_sub3 = 'data_preprocessed/epochs_emg_train_sub3.npz'
	eeg1_train_sub3 = np.load(path_eeg1_train_sub3)['arr_0']
	eeg2_train_sub3 = np.load(path_eeg2_train_sub3)['arr_0']
	emg_train_sub3 = np.load(path_emg_train_sub3)['arr_0']

	path_label_train_sub3 = 'data_preprocessed/labels_train_sub3.npz'
	label_train_sub3 = np.load(path_label_train_sub3)['arr_0'] - 1		# to match with to_categorical()
	
	# reshape & expand dim
	for sig in ['eeg1', 'eeg2', 'emg']:
		for sub in ['1','2','3']:
			vars()[sig+'_train_sub'+sub] = vars()[sig+'_train_sub'+sub].reshape(-1, 160, 48, 1)

	# combine train data to 3 channels
	for sub in ['1','2','3']:
		vars()['input_train_sub'+sub] = np.concatenate((vars()['eeg1_train_sub'+sub],vars()['eeg2_train_sub'+sub],vars()['emg_train_sub'+sub]),
			axis=3)

	# labels to categorical
	#label_train_sub1_onehot = to_categorical(label_train_sub1)
	#print(label_train_sub1_onehot.shape)
	#label_train_sub2_onehot = to_categorical(label_train_sub2)
	#label_train_sub3_onehot = to_categorical(label_train_sub3)


	# combine train data
	input_train = np.concatenate((input_train_sub1, input_train_sub2),axis=0)
	#label_train = np.concatenate((label_train_sub1_onehot, label_train_sub2_onehot),axis=0)
	label_train = np.concatenate((label_train_sub1, label_train_sub2),axis=0)

	# shuffle train & val data
	input_train, label_train = unison_shuffle(input_train, label_train)
	#print(input_train.shape)
	#print(label_train)
	#input_val, label_val = unison_shuffle(input_train_sub3, label_train_sub3_onehot)
	input_val, label_val = unison_shuffle(input_train_sub3, label_train_sub3)

	# CNN architecture
	model = Sequential([
		MaxPooling2D(pool_size=(2,6),strides=(2,6),input_shape=(160,48,3)),
		Conv2D(50,kernel_size=(3,3),strides=(1,1),activation='relu'),
		MaxPooling2D(pool_size=(2,2),strides=(2,2)),
		Flatten(),
		Dense(1000, kernel_initializer="lecun_uniform"),
		Dropout(.5),
		Dense(3, kernel_initializer="lecun_uniform"),
		Softmax(),
		])

	# CNN train
	model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=5e-5), 
		metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
	class_weight = {0:1/34114, 1:1/27133, 2:1/3553}    #stat: 1:34114, 2:27133, 3:3553   total:64800
	print(label_train.shape)
	train = model.fit(input_train, label_train, 
		class_weight=class_weight, batch_size=100, epochs=1, verbose=2, 
		validation_split=0.1)
	

	# CNN prediction
	val_pred = model.predict(input_val, verbose=0)
	label_val_pred = np.argmax(np.round(val_pred),axis=1)

	# save model
	model.save('model_12.h5')
	np.save('label_pred_3.npy',label_val_pred)

	print(label_val)
	print(label_val_pred)
	# metric
	BMAC = balanced_accuracy_score(label_val, label_val_pred)
	
	# print result
	print(BMAC)