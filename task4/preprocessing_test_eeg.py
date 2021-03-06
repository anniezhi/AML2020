## AML Task 4: Sleep Stage Classification ##
# Test EEG data
# Preprocessing: Fourier transform, PSD, bandpass, (energy), log, per-freq standardization, epoch split, neighbor concat

import numpy as np
import pandas as pd
from numpy import genfromtxt
from scipy.signal import stft
from numpy import savez_compressed

def import_data(path):
	data = genfromtxt(path, delimiter=',')
	data = data[1:,1:]
	return data

def subject_sep(data):
	num_epochs_per_sub = 21600
	sub1 = data[:num_epochs_per_sub,:]
	sub2 = data[num_epochs_per_sub:,:]
	return sub1, sub2

def epoch_sep(data, len_chunk):
	for i in range(0, data.shape[1]-1, len_chunk):
		yield data[:,i:i+len_chunk]

def epoch_concat(data):
	for i in range(len(data)):
		if i<2:
			yield np.concatenate(np.concatenate([np.zeros([2-i,48,32]),data[:i+3,:,:]],axis=0), axis=1)
		elif i>21597:
			yield np.concatenate(np.concatenate([data[i-2:,:,:],np.zeros([i-21597,48,32])],axis=0), axis=1)
		else:
			yield np.concatenate(data[i-2:i+3,:,:], axis=1)

if __name__ == '__main__':

	# import data
	path_eeg1_test = 'data/test_eeg1.csv'
	eeg1_test_org = import_data(path_eeg1_test)
	path_eeg2_test = 'data/test_eeg2.csv'
	eeg2_test_org = import_data(path_eeg2_test)

# subject separation
	## Test
	eeg1_test_sub1, eeg1_test_sub2 = subject_sep(eeg1_test_org)
	eeg2_test_sub1, eeg2_test_sub2 = subject_sep(eeg2_test_org)
	

	# per-subject signal combination
	#e.g. eeg1_train_sub1_full = np.concatenate(eeg1_train_sub1)
	## Test
	for sig in ['eeg1', 'eeg2']:
		for sub in ['1', '2']:
			vars()[sig+'_test_sub'+sub+'_full'] = np.concatenate(vars()[sig+'_test_sub'+sub])


	# Fourier Transform
	#e.g. ft_emg_train_sub1 = stft(emg_train_sub1_full, fs=128, window="hamming", nperseg=256, noverlap=256-16)
	#output = (y_axis_freq,x_axis_time,ft_complex)
	## Test
	for sig in ['eeg1', 'eeg2']:
		for sub in ['1', '2']:
			vars()['ft_'+sig+'_test_sub'+sub] = stft(vars()[sig+'_test_sub'+sub+'_full'], fs=128,
													 window='hamming', nperseg=256, noverlap=256-16)

	# PSD - amplitude, bandpass, (energy), log, (per-freq) standardization
	#e.g. step 1: psd_{eeg1,eeg2,emg}_train_sub1 = abs(ft_{eeg1,eeg2,emg}_train_sub1[2])**2
	#     step 2: psd_bp_{eeg1,eeg2}_train_sub1 = psd_{eeg1,eeg2}_train_sub1[1:49,:]
	#			  psd_bp_{emg}_train_sub1 = psd_{emg}_train_sub1[1:61,:]
	#	  step 3: energy_{emg}_train_sub1 = np.sum(psd_bp_{emg}_train_sub1, axis=0)
	#	  step 4: log_{eeg1,eeg2}_train_sub1 = np.log(psd_bp_{eeg1,eeg2}_train_sub1)
	#			  log_{emg}_train_sub1 = np.log(energy_{emg}_train_sub1)
	#	  step 5: std_log_{eeg1,eeg2}_train_sub1 = (log_{eeg1,eeg2}_train_sub1 - np.mean(log_{eeg1,eeg2}_train_sub1,axis=1)[:,None]) / np.std(log_{eeg1,eeg2}_train_sub1,axis=1)[:,None]
	#	  		  std_log_{emg}_train_sub1 = (log_{emg}_train_sub1 - np.mean(log_{emg}_train_sub1)) / np.std(log_{emg}_train_sub1)
	## Test
	for sig in ['eeg1', 'eeg2']:
		for sub in ['1', '2']:
			### step 1
			vars()['psd_'+sig+'_test_sub'+sub] = abs(vars()['ft_'+sig+'_test_sub'+sub][2])**2
			### step 2
			vars()['psd_bp_'+sig+'_test_sub'+sub] = vars()['psd_'+sig+'_test_sub'+sub][1:49,:]
			### step 4
			vars()['log_'+sig+'_test_sub'+sub] = np.log(vars()['psd_bp_'+sig+'_test_sub'+sub])
			### step 5
			vars()['std_log_'+sig+'_test_sub'+sub] = (vars()['log_'+sig+'_test_sub'+sub] - np.mean(vars()['log_'+sig+'_test_sub'+sub],axis=1)[:,None]) / np.std(vars()['log_'+sig+'_test_sub'+sub],axis=1)[:,None]
			

	# epoch split to 4-sec
	#e.g. epoch_single_ = list(epoch_sep(std_log_emg_test_sub1,32))
	#output shape: (21600, 48, 32)
	## Test
	for sig in ['eeg1', 'eeg2']:
		for sub in ['1', '2']:
			vars()['epoch_single_'+sig+'_test_sub'+sub] = np.array(list(epoch_sep(vars()['std_log_'+sig+'_test_sub'+sub], 32)))

	
	# epoch concatenation to 5 as a group
	#e.g. epochs_ = epoch_concat(epoch_single_eeg1_train_sub1)
	#output shape: (21600, 48, 5*32)
	## Test
	for sig in ['eeg1', 'eeg2']:
		for sub in ['1', '2']:
			vars()['epochs_'+sig+'_test_sub'+sub] = np.array(list(epoch_concat(vars()['epoch_single_'+sig+'_test_sub'+sub])))


	# save epochs to npz
	## Test
	for sig in ['eeg1', 'eeg2']:
		for sub in ['1', '2']:
			savez_compressed('epochs_'+sig+'_test_sub'+sub+'.npz',varss()['epochs_'+sig+'_test_sub'+sub])