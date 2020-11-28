## AML Task 3: ECG Classification ##
# Preprocessing: scaling, feature extraction

import numpy as np
import pandas as pd
from numpy import genfromtxt
import biosppy.signals.ecg as ecg
import neurokit2 as nk
from tqdm import tqdm


def import_data(path):
	data = genfromtxt(path, delimiter=',')
	data = data[1:,1:]
	return data

def scaling_standard(X):
	X_mean = np.expand_dims(np.nanmean(X, axis=1),1)
	X_std = np.expand_dims(np.nanstd(X, axis=1),1)
	X_scaled = (X - X_mean) / X_std
	return X_scaled

def scaling_minmax(X):
	X_min = np.expand_dims(np.nanmin(X, axis=1),1)
	X_max = np.expand_dims(np.nanmax(X, axis=1),1)
	X_scaled = 2*(X-X_min)/(X_max-X_min) - 1
	return X_scaled

if __name__ == '__main__':

	# import data
	path_X_train = 'data/X_train.csv'
	X_train_org = import_data(path_X_train)
	path_y_train = 'data/y_train.csv'
	y_train_org = import_data(path_y_train).ravel()
	path_X_test = 'data/X_test.csv'
	X_test_org = import_data(path_X_test)
	#print(X_train_org.shape)

	# scaling - train
	#X_train_scaled_standard = scaling_standard(X_train_org)
	#X_train_scaled_minmax = scaling_minmax(X_train_org)

	# scaling - test
	#X_test_scaled_standard = scaling_standard(X_test_org)
	#X_test_scaled_minmax = scaling_minmax(X_test_org)

	# select scaler
	## standard
	#X_train = X_train_scaled_standard
	#X_test = X_test_scaled_standard

	## minmax
	#X_train = X_train_scaled_minmax
	#X_test = X_test_scaled_minmax
	
	## no scaling
	X_train = X_train_org
	X_test = X_test_org


	# feature extraction - train
	features_all_train = []

	for i in tqdm(range(len(X_train))):
		X = X_train[i,:][~np.isnan(X_train[i,:])]

		# feature: sequence length
		time_len = len(X) / 300
		
		ts, filtered_biosppy, rpeaks_biosppy, _, _, _, heart_rate = ecg.ecg(X, sampling_rate=300, show=False)
		
		# feature: r peaks
		r_peaks_amp = filtered_biosppy[rpeaks_biosppy]
		if r_peaks_amp.size > 0:
			r_peaks_min = min(r_peaks_amp)
			r_peaks_max = max(r_peaks_amp)
			r_peaks_mean = np.mean(r_peaks_amp)
			r_peaks_std = np.std(r_peaks_amp)
			r_peaks_median = np.median(r_peaks_amp)
		else:
			r_peaks_min, r_peaks_max, r_peaks_mean, r_peaks_std, r_peaks_median = [None] * 5

		# feature: rr intervals
		rr_intervals = ts[rpeaks_biosppy[1:]] - ts[rpeaks_biosppy[:-1]]
		if rr_intervals.size > 0:
			rr_intervals_min = min(rr_intervals)
			rr_intervals_max = max(rr_intervals)
			rr_intervals_mean = np.mean(rr_intervals)
			rr_intervals_std = np.std(rr_intervals)
			rr_intervals_median = np.median(rr_intervals)
		else:
			rr_intervals_min, rr_intervals_max, rr_intervals_mean, rr_intervals_std, rr_intervals_median = [None] * 5

		# feature: heart rate
		if heart_rate.size > 0:
			heart_rate_min = min(heart_rate)
			heart_rate_max = max(heart_rate)
			heart_rate_mean = np.mean(heart_rate)
			heart_rate_std = np.std(heart_rate)
			heart_rate_median = np.median(heart_rate)
		else:
			heart_rate_min, heart_rate_max, heart_rate_mean, heart_rate_std, heart_rate_median = [None] * 5

		X_cleaned_neurokit = nk.ecg_clean(X, sampling_rate=300)
		rpeaks_neurokit_onehot, rpeaks_neurokit = nk.ecg_peaks(X_cleaned_neurokit, sampling_rate=300)
		#print(i)
		if len(rpeaks_neurokit['ECG_R_Peaks']) <= 10:
			features = [time_len, 
					r_peaks_min, r_peaks_max, r_peaks_mean, r_peaks_std, r_peaks_median,
					rr_intervals_min, rr_intervals_max, rr_intervals_mean, rr_intervals_std, rr_intervals_median,
					heart_rate_min, heart_rate_max, heart_rate_mean, heart_rate_std, heart_rate_median]
			features.extend([None]*78)
			features_all_train.append(features)
			continue
		_, waves_peaks_neurokit = nk.ecg_delineate(X_cleaned_neurokit, rpeaks_neurokit, sampling_rate=300)

		# feature: t peaks
		tpeaks_neurokit = np.array(waves_peaks_neurokit['ECG_T_Peaks'])
		t_peaks_amp = X_cleaned_neurokit[tpeaks_neurokit[~np.isnan(tpeaks_neurokit)].astype(int)]
		if t_peaks_amp.size > 0:
			t_peaks_min = min(t_peaks_amp)
			t_peaks_max = max(t_peaks_amp)
			t_peaks_mean = np.mean(t_peaks_amp)
			t_peaks_std = np.std(t_peaks_amp)
			t_peaks_median = np.median(t_peaks_amp)
		else:
			t_peaks_min, t_peaks_max, t_peaks_mean, t_peaks_std, t_peaks_median = [None] * 5

		# feature: p peaks
		ppeaks_neurokit = np.array(waves_peaks_neurokit['ECG_P_Peaks'])
		p_peaks_amp = X_cleaned_neurokit[ppeaks_neurokit[~np.isnan(ppeaks_neurokit)].astype(int)]
		if p_peaks_amp.size > 0:
			p_peaks_min = min(p_peaks_amp)
			p_peaks_max = max(p_peaks_amp)
			p_peaks_mean = np.mean(p_peaks_amp)
			p_peaks_std = np.std(p_peaks_amp)
			p_peaks_median = np.median(p_peaks_amp)
		else:
			p_peaks_min, p_peaks_max, p_peaks_mean, p_peaks_std, p_peaks_median = [None] * 5

		# feature: q peaks
		qpeaks_neurokit = np.array(waves_peaks_neurokit['ECG_Q_Peaks'])
		q_peaks_amp = X_cleaned_neurokit[qpeaks_neurokit[~np.isnan(qpeaks_neurokit)].astype(int)]
		if q_peaks_amp.size > 0:
			q_peaks_min = min(q_peaks_amp)
			q_peaks_max = max(q_peaks_amp)
			q_peaks_mean = np.mean(q_peaks_amp)
			q_peaks_std = np.std(q_peaks_amp)
			q_peaks_median = np.median(q_peaks_amp)
		else:
			q_peaks_min, q_peaks_max, q_peaks_mean, q_peaks_std, q_peaks_median = [None] * 5

		# feature: s peaks
		speaks_neurokit = np.array(waves_peaks_neurokit['ECG_S_Peaks'])
		s_peaks_amp = X_cleaned_neurokit[speaks_neurokit[~np.isnan(speaks_neurokit)].astype(int)]
		if s_peaks_amp.size > 0:
			s_peaks_min = min(s_peaks_amp)
			s_peaks_max = max(s_peaks_amp)
			s_peaks_mean = np.mean(s_peaks_amp)
			s_peaks_std = np.std(s_peaks_amp)
			s_peaks_median = np.median(s_peaks_amp)
		else:
			s_peaks_min, s_peaks_max, s_peaks_mean, s_peaks_std, s_peaks_median = [None] * 5

		# feature: qrs duration
		qrs_duration = ts[speaks_neurokit[~np.isnan(speaks_neurokit-qpeaks_neurokit)].astype(int)] - ts[qpeaks_neurokit[~np.isnan(speaks_neurokit-qpeaks_neurokit)].astype(int)]
		if qrs_duration.size > 0:
			qrs_duration_min = min(qrs_duration)
			qrs_duration_max = max(qrs_duration)
			qrs_duration_mean = np.mean(qrs_duration)
			qrs_duration_std = np.std(qrs_duration)
			qrs_duration_median = np.median(qrs_duration)
		else:
			qrs_duration_min, qrs_duration_max, qrs_duration_mean, qrs_duration_std, qrs_duration_median = [None] * 5
		
		# feature: hrv
		hrv = nk.hrv(peaks=rpeaks_neurokit_onehot, show=False)
		for column in hrv.columns:
			vars()[column.lower()] = hrv[column].iloc[0]
		#hrv_rmssd, hrv_meannn, hrv_sdnn, hrv_sdsd, hrv_cvnn, hrv_cvsd, hrv_mediannn
		#hrv_madnn, hrv_mcvnn, hrv_iqrnn, hrv_pnn50, hrv_pnn20, hrv_tinn, hrv_hti
		#hrv_ulf, hrv_vlf, hrv_lf, hrv_hf, hrv_vhf, hrv_lfhf, hrv_lfn, hrv_hfn, hrv_lnhf
		#hrv_sd1, hrv_sd2, hrv_sd1sd2, hrv_s, hrv_csi, hrv_cvi, hrv_csi_modified, hrv_pip
		#hrv_ials, hrv_pss, hrv_pas, hrv_gi, hrv_si, hrv_ai, hrv_pi, hrv_c1d, hrv_c1a
		#hrv_sd1d, hrv_sd1a, hrv_c2d, hrv_c2a, hrv_sd2d, hrv_sd2a, hrv_cd, hrv_ca, hrv_sdnnd
		#hrv_sdnna, hrv_apen, hrv_sampen

		# feature: ecg quality
		X_processed, _ = nk.ecg_process(X, sampling_rate=300)
		ecg_quality_avg = X_processed['ECG_Quality'].sum() / len(X)

		# feature combination
		features = [time_len, 
					r_peaks_min, r_peaks_max, r_peaks_mean, r_peaks_std, r_peaks_median,
					rr_intervals_min, rr_intervals_max, rr_intervals_mean, rr_intervals_std, rr_intervals_median,
					heart_rate_min, heart_rate_max, heart_rate_mean, heart_rate_std, heart_rate_median,
					t_peaks_min, t_peaks_max, t_peaks_mean, t_peaks_std, t_peaks_median,
					p_peaks_min, p_peaks_max, p_peaks_mean, p_peaks_std, p_peaks_median,
					q_peaks_min, q_peaks_max, q_peaks_mean, q_peaks_std, q_peaks_median,
					s_peaks_min, s_peaks_max, s_peaks_mean, s_peaks_std, s_peaks_median,
					qrs_duration_min, qrs_duration_max, qrs_duration_mean, qrs_duration_std, qrs_duration_median,
					hrv_rmssd, hrv_meannn, hrv_sdnn, hrv_sdsd, hrv_cvnn, hrv_cvsd, hrv_mediannn,
					hrv_madnn, hrv_mcvnn, hrv_iqrnn, hrv_pnn50, hrv_pnn20, hrv_tinn, hrv_hti,
					hrv_ulf, hrv_vlf, hrv_lf, hrv_hf, hrv_vhf, hrv_lfhf, hrv_lfn, hrv_hfn, hrv_lnhf,
					hrv_sd1, hrv_sd2, hrv_sd1sd2, hrv_s, hrv_csi, hrv_cvi, hrv_csi_modified, hrv_pip,
					hrv_ials, hrv_pss, hrv_pas, hrv_gi, hrv_si, hrv_ai, hrv_pi, hrv_c1d, hrv_c1a,
					hrv_sd1d, hrv_sd1a, hrv_c2d, hrv_c2a, hrv_sd2d, hrv_sd2a, hrv_cd, hrv_ca, hrv_sdnnd,
					hrv_sdnna, hrv_apen, hrv_sampen,
					ecg_quality_avg]

		# feature append
		features_all_train.append(features)


	# feature extraction - test
	features_all_test = []

	for i in tqdm(range(len(X_test))):
		X = X_test[i,:][~np.isnan(X_test[i,:])]

		# feature: sequence length
		time_len = len(X) / 300
		
		ts, filtered_biosppy, rpeaks_biosppy, _, _, _, heart_rate = ecg.ecg(X, sampling_rate=300, show=False)
		
		# feature: r peaks
		r_peaks_amp = filtered_biosppy[rpeaks_biosppy]
		if r_peaks_amp.size > 0:
			r_peaks_min = min(r_peaks_amp)
			r_peaks_max = max(r_peaks_amp)
			r_peaks_mean = np.mean(r_peaks_amp)
			r_peaks_std = np.std(r_peaks_amp)
			r_peaks_median = np.median(r_peaks_amp)
		else:
			r_peaks_min, r_peaks_max, r_peaks_mean, r_peaks_std, r_peaks_median = [None] * 5

		# feature: rr intervals
		rr_intervals = ts[rpeaks_biosppy[1:]] - ts[rpeaks_biosppy[:-1]]
		if rr_intervals.size > 0:
			rr_intervals_min = min(rr_intervals)
			rr_intervals_max = max(rr_intervals)
			rr_intervals_mean = np.mean(rr_intervals)
			rr_intervals_std = np.std(rr_intervals)
			rr_intervals_median = np.median(rr_intervals)
		else:
			rr_intervals_min, rr_intervals_max, rr_intervals_mean, rr_intervals_std, rr_intervals_median = [None] * 5

		# feature: heart rate
		if heart_rate.size > 0:
			heart_rate_min = min(heart_rate)
			heart_rate_max = max(heart_rate)
			heart_rate_mean = np.mean(heart_rate)
			heart_rate_std = np.std(heart_rate)
			heart_rate_median = np.median(heart_rate)
		else:
			heart_rate_min, heart_rate_max, heart_rate_mean, heart_rate_std, heart_rate_median = [None] * 5

		X_cleaned_neurokit = nk.ecg_clean(X, sampling_rate=300)
		rpeaks_neurokit_onehot, rpeaks_neurokit = nk.ecg_peaks(X_cleaned_neurokit, sampling_rate=300)
		print(i)
		if len(rpeaks_neurokit['ECG_R_Peaks']) <= 10:
			features = [time_len, 
					r_peaks_min, r_peaks_max, r_peaks_mean, r_peaks_std, r_peaks_median,
					rr_intervals_min, rr_intervals_max, rr_intervals_mean, rr_intervals_std, rr_intervals_median,
					heart_rate_min, heart_rate_max, heart_rate_mean, heart_rate_std, heart_rate_median]
			features.extend([None]*78)
			features_all_test.append(features)
			continue
		_, waves_peaks_neurokit = nk.ecg_delineate(X_cleaned_neurokit, rpeaks_neurokit, sampling_rate=300)

		# feature: t peaks
		tpeaks_neurokit = np.array(waves_peaks_neurokit['ECG_T_Peaks'])
		t_peaks_amp = X_cleaned_neurokit[tpeaks_neurokit[~np.isnan(tpeaks_neurokit)].astype(int)]
		if t_peaks_amp.size > 0:
			t_peaks_min = min(t_peaks_amp)
			t_peaks_max = max(t_peaks_amp)
			t_peaks_mean = np.mean(t_peaks_amp)
			t_peaks_std = np.std(t_peaks_amp)
			t_peaks_median = np.median(t_peaks_amp)
		else:
			t_peaks_min, t_peaks_max, t_peaks_mean, t_peaks_std, t_peaks_median = [None] * 5

		# feature: p peaks
		ppeaks_neurokit = np.array(waves_peaks_neurokit['ECG_P_Peaks'])
		p_peaks_amp = X_cleaned_neurokit[ppeaks_neurokit[~np.isnan(ppeaks_neurokit)].astype(int)]
		if p_peaks_amp.size > 0:
			p_peaks_min = min(p_peaks_amp)
			p_peaks_max = max(p_peaks_amp)
			p_peaks_mean = np.mean(p_peaks_amp)
			p_peaks_std = np.std(p_peaks_amp)
			p_peaks_median = np.median(p_peaks_amp)
		else:
			p_peaks_min, p_peaks_max, p_peaks_mean, p_peaks_std, p_peaks_median = [None] * 5

		# feature: q peaks
		qpeaks_neurokit = np.array(waves_peaks_neurokit['ECG_Q_Peaks'])
		q_peaks_amp = X_cleaned_neurokit[qpeaks_neurokit[~np.isnan(qpeaks_neurokit)].astype(int)]
		if q_peaks_amp.size > 0:
			q_peaks_min = min(q_peaks_amp)
			q_peaks_max = max(q_peaks_amp)
			q_peaks_mean = np.mean(q_peaks_amp)
			q_peaks_std = np.std(q_peaks_amp)
			q_peaks_median = np.median(q_peaks_amp)
		else:
			q_peaks_min, q_peaks_max, q_peaks_mean, q_peaks_std, q_peaks_median = [None] * 5

		# feature: s peaks
		speaks_neurokit = np.array(waves_peaks_neurokit['ECG_S_Peaks'])
		s_peaks_amp = X_cleaned_neurokit[speaks_neurokit[~np.isnan(speaks_neurokit)].astype(int)]
		if s_peaks_amp.size > 0:
			s_peaks_min = min(s_peaks_amp)
			s_peaks_max = max(s_peaks_amp)
			s_peaks_mean = np.mean(s_peaks_amp)
			s_peaks_std = np.std(s_peaks_amp)
			s_peaks_median = np.median(s_peaks_amp)
		else:
			s_peaks_min, s_peaks_max, s_peaks_mean, s_peaks_std, s_peaks_median = [None] * 5

		# feature: qrs duration
		qrs_duration = ts[speaks_neurokit[~np.isnan(speaks_neurokit-qpeaks_neurokit)].astype(int)] - ts[qpeaks_neurokit[~np.isnan(speaks_neurokit-qpeaks_neurokit)].astype(int)]
		if qrs_duration.size > 0:
			qrs_duration_min = min(qrs_duration)
			qrs_duration_max = max(qrs_duration)
			qrs_duration_mean = np.mean(qrs_duration)
			qrs_duration_std = np.std(qrs_duration)
			qrs_duration_median = np.median(qrs_duration)
		else:
			qrs_duration_min, qrs_duration_max, qrs_duration_mean, qrs_duration_std, qrs_duration_median = [None] * 5

		# feature: hrv
		hrv = nk.hrv(peaks=rpeaks_neurokit_onehot, show=False)
		for column in hrv.columns:
			vars()[column.lower()] = hrv[column].iloc[0]
		#hrv_rmssd, hrv_meannn, hrv_sdnn, hrv_sdsd, hrv_cvnn, hrv_cvsd, hrv_mediannn
		#hrv_madnn, hrv_mcvnn, hrv_iqrnn, hrv_pnn50, hrv_pnn20, hrv_tinn, hrv_hti
		#hrv_ulf, hrv_vlf, hrv_lf, hrv_hf, hrv_vhf, hrv_lfhf, hrv_lfn, hrv_hfn, hrv_lnhf
		#hrv_sd1, hrv_sd2, hrv_sd1sd2, hrv_s, hrv_csi, hrv_cvi, hrv_csi_modified, hrv_pip
		#hrv_ials, hrv_pss, hrv_pas, hrv_gi, hrv_si, hrv_ai, hrv_pi, hrv_c1d, hrv_c1a
		#hrv_sd1d, hrv_sd1a, hrv_c2d, hrv_c2a, hrv_sd2d, hrv_sd2a, hrv_cd, hrv_ca, hrv_sdnnd
		#hrv_sdnna, hrv_apen, hrv_sampen

		# feature: ecg quality
		X_processed, _ = nk.ecg_process(X, sampling_rate=300)
		ecg_quality_avg = X_processed['ECG_Quality'].sum() / len(X)

		# feature combination
		features = [time_len, 
					r_peaks_min, r_peaks_max, r_peaks_mean, r_peaks_std, r_peaks_median,
					rr_intervals_min, rr_intervals_max, rr_intervals_mean, rr_intervals_std, rr_intervals_median,
					heart_rate_min, heart_rate_max, heart_rate_mean, heart_rate_std, heart_rate_median,
					t_peaks_min, t_peaks_max, t_peaks_mean, t_peaks_std, t_peaks_median,
					p_peaks_min, p_peaks_max, p_peaks_mean, p_peaks_std, p_peaks_median,
					q_peaks_min, q_peaks_max, q_peaks_mean, q_peaks_std, q_peaks_median,
					s_peaks_min, s_peaks_max, s_peaks_mean, s_peaks_std, s_peaks_median,
					qrs_duration_min, qrs_duration_max, qrs_duration_mean, qrs_duration_std, qrs_duration_median,
					hrv_rmssd, hrv_meannn, hrv_sdnn, hrv_sdsd, hrv_cvnn, hrv_cvsd, hrv_mediannn,
					hrv_madnn, hrv_mcvnn, hrv_iqrnn, hrv_pnn50, hrv_pnn20, hrv_tinn, hrv_hti,
					hrv_ulf, hrv_vlf, hrv_lf, hrv_hf, hrv_vhf, hrv_lfhf, hrv_lfn, hrv_hfn, hrv_lnhf,
					hrv_sd1, hrv_sd2, hrv_sd1sd2, hrv_s, hrv_csi, hrv_cvi, hrv_csi_modified, hrv_pip,
					hrv_ials, hrv_pss, hrv_pas, hrv_gi, hrv_si, hrv_ai, hrv_pi, hrv_c1d, hrv_c1a,
					hrv_sd1d, hrv_sd1a, hrv_c2d, hrv_c2a, hrv_sd2d, hrv_sd2a, hrv_cd, hrv_ca, hrv_sdnnd,
					hrv_sdnna, hrv_apen, hrv_sampen,
					ecg_quality_avg]

		# feature append
		features_all_test.append(features)


	# save features to dataframe
	features_all_names = ['time_len', 
				  		  'r_peaks_min', 'r_peaks_max', 'r_peaks_mean', 'r_peaks_std', 'r_peaks_median',
				  		  'rr_intervals_min', 'rr_intervals_max', 'rr_intervals_mean', 'rr_intervals_std', 'rr_intervals_median',
				  		  'heart_rate_min', 'heart_rate_max', 'heart_rate_mean', 'heart_rate_std', 'heart_rate_median',
				  		  't_peaks_min', 't_peaks_max', 't_peaks_mean', 't_peaks_std', 't_peaks_median',
				  		  'p_peaks_min', 'p_peaks_max', 'p_peaks_mean', 'p_peaks_std', 'p_peaks_median',
				  		  'q_peaks_min', 'q_peaks_max', 'q_peaks_mean', 'q_peaks_std', 'q_peaks_median',
				  		  's_peaks_min', 's_peaks_max', 's_peaks_mean', 's_peaks_std', 's_peaks_median',
				  		  'qrs_duration_min', 'qrs_duration_max', 'qrs_duration_mean', 'qrs_duration_std', 'qrs_duration_median',
				  		  'hrv_rmssd', 'hrv_meannn', 'hrv_sdnn', 'hrv_sdsd', 'hrv_cvnn', 'hrv_cvsd', 'hrv_mediannn',
						  'hrv_madnn', 'hrv_mcvnn', 'hrv_iqrnn', 'hrv_pnn50', 'hrv_pnn20', 'hrv_tinn', 'hrv_hti',
						  'hrv_ulf', 'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_vhf', 'hrv_lfhf', 'hrv_lfn', 'hrv_hfn', 'hrv_lnhf',
						  'hrv_sd1', 'hrv_sd2', 'hrv_sd1sd2', 'hrv_s', 'hrv_csi', 'hrv_cvi', 'hrv_csi_modified', 'hrv_pip',
						  'hrv_ials', 'hrv_pss', 'hrv_pas', 'hrv_gi', 'hrv_si', 'hrv_ai', 'hrv_pi', 'hrv_c1d', 'hrv_c1a',
						  'hrv_sd1d', 'hrv_sd1a', 'hrv_c2d', 'hrv_c2a', 'hrv_sd2d', 'hrv_sd2a', 'hrv_cd', 'hrv_ca', 'hrv_sdnnd',
						  'hrv_sdnna', 'hrv_apen', 'hrv_sampen',
				  		  'ecg_quality_avg']

	with open('features_all_train_org.txt','w') as f:
		for _list in features_all_train:
			for _string in _list:
				f.write(str(_string) + '\n')

	df_train = pd.DataFrame(features_all_train)
	df_train.columns = features_all_names

	df_test = pd.DataFrame(features_all_test)
	df_test.columns = features_all_names


	df_train.to_csv('X_train_features.csv',index=True)
	df_test.to_csv('X_test_features.csv',index=True)