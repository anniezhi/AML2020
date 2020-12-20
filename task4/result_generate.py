## AML Task 4: Sleep Stage Classification ##
# Write results to file as required

import numpy as np
import pandas as pd
from numpy import genfromtxt

if __name__ == '__main__':
	# construct samples
	output_sample = pd.read_csv('data/sample.csv')
	output_model_12 = output_sample
	output_model_13 = output_sample
	output_model_23 = output_sample
	output_model_all = output_sample

	# read predictions
	label_pred_model_12 = np.load('label_pred_test_model_12.npy')
	label_pred_model_13 = np.load('label_pred_test_model_13.npy')
	label_pred_model_23 = np.load('label_pred_test_model_23.npy')
	label_pred_model_all = np.load('label_pred_test_model_all.npy')

	# assign label values
	output_model_12['y'] = label_pred_model_12.astype(int)
	output_model_23['y'] = label_pred_model_23.astype(int)
	output_model_13['y'] = label_pred_model_13.astype(int)
	output_model_all['y'] = label_pred_model_all.astype(int)

	# save output files
	output_model_12.to_csv('output_model_12.csv', index=False)
	output_model_13.to_csv('output_model_13.csv', index=False)
	output_model_23.to_csv('output_model_23.csv', index=False)
	output_model_all.to_csv('output_model_all.csv', index=False)