import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.eeg as eeg
import biosppy.signals.emg as emg
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score


def splitData(X_train, y_train):
    amountOfOneSample = X_train.shape[0] // 3
    randInt = np.random.randint(3)
    testing_data = X_train[randInt*amountOfOneSample:(randInt+1)*amountOfOneSample]
    training_data = X_train[:randInt*amountOfOneSample]
    training_data = np.append(training_data, X_train[(randInt+1)*amountOfOneSample:], axis=0)
    
    testing_y = y_train[randInt*amountOfOneSample:(randInt+1)*amountOfOneSample]
    training_y = y_train[:randInt*amountOfOneSample]
    training_y = np.append(training_y, y_train[(randInt+1)*amountOfOneSample:])
    return training_data, training_y, testing_data, testing_y


if __name__ == '__main__':

    # Load Data
    train_eeg_1 = pd.read_csv('train_eeg1.csv', index_col=0)
    train_eeg_2 = pd.read_csv('train_eeg2.csv', index_col=0)
    train_emg = pd.read_csv('train_emg.csv', index_col=0)
    y_train = pd.read_csv('train_labels.csv', index_col=0)

    # Load test data
    test_eeg_1 = pd.read_csv('test_eeg1.csv', index_col=0)
    test_eeg_2 = pd.read_csv('test_eeg2.csv', index_col=0)
    test_emg = pd.read_csv('test_emg.csv', index_col=0)

    # Convert to numpy arrays
    X_eeg_1 = train_eeg_1.to_numpy()
    X_eeg_2 = train_eeg_2.to_numpy()
    X_emg = train_emg.to_numpy()
    y = y_train.to_numpy()
    # test data
    X_test_eeg_1 = test_eeg_1.to_numpy()
    X_test_eeg_2 = test_eeg_2.to_numpy()
    X_test_emg = test_emg.to_numpy()

    # separate training subjects
    amountOfSubjectSamples = 21600
    eeg_subject_1_ch_1 = X_eeg_1[:amountOfSubjectSamples]
    eeg_subject_1_ch_2 = X_eeg_2[:amountOfSubjectSamples]
    eeg_subject_2_ch_1 = X_eeg_1[amountOfSubjectSamples:2*amountOfSubjectSamples]
    eeg_subject_2_ch_2 = X_eeg_2[amountOfSubjectSamples:2*amountOfSubjectSamples]
    eeg_subject_3_ch_1 = X_eeg_1[2*amountOfSubjectSamples:]
    eeg_subject_3_ch_2 = X_eeg_2[2*amountOfSubjectSamples:]
    emg_subject_1 = X_emg[:amountOfSubjectSamples]
    emg_subject_2 = X_emg[amountOfSubjectSamples:2*amountOfSubjectSamples]
    emg_subject_3 = X_emg[2*amountOfSubjectSamples:]

    # test data
    eeg_test_1_ch_1 = X_test_eeg_1[:amountOfSubjectSamples]
    eeg_test_1_ch_2 = X_test_eeg_2[:amountOfSubjectSamples]
    eeg_test_2_ch_1 = X_test_eeg_1[amountOfSubjectSamples:2*amountOfSubjectSamples]
    eeg_test_2_ch_2 = X_test_eeg_2[amountOfSubjectSamples:2*amountOfSubjectSamples]
    emg_test_1 = X_test_emg[:amountOfSubjectSamples]
    emg_test_2 = X_test_emg[amountOfSubjectSamples:2*amountOfSubjectSamples]


    # represent eeg channels in matrices
    eeg_subject_1 = np.column_stack([eeg_subject_1_ch_1.ravel(), eeg_subject_1_ch_2.ravel()])
    eeg_subject_2 = np.column_stack([eeg_subject_2_ch_1.ravel(), eeg_subject_2_ch_2.ravel()])
    eeg_subject_3 = np.column_stack([eeg_subject_3_ch_1.ravel(), eeg_subject_3_ch_2.ravel()])
    # test data
    eeg_test_1 = np.column_stack([eeg_test_1_ch_1.ravel(), eeg_test_1_ch_2.ravel()])
    eeg_test_2 = np.column_stack([eeg_test_2_ch_1.ravel(), eeg_test_2_ch_2.ravel()])

    # extract frequencies, takes long to compute, SAVE IT AND THEN LOAD IT
    subject_1_freqs = eeg.eeg(eeg_subject_1, sampling_rate=128, show=False)
    subject_2_freqs = eeg.eeg(eeg_subject_2, sampling_rate=128, show=False)
    subject_3_freqs = eeg.eeg(eeg_subject_3, sampling_rate=128, show=False)
    # test data
    test_subject_1_freqs = eeg.eeg(eeg_test_1, sampling_rate=128, show=False)
    test_subject_2_freqs = eeg.eeg(eeg_test_2, sampling_rate=128, show=False)

    # save the data, do not recompute it
    names = ['theta', 'alpha_low', 'alpha_high', 'beta', 'gamma']
    #for j, subject in enumerate([subject_1_freqs, subject_2_freqs, subject_3_freqs]):
    #    for i in range(3,8):
    #        np.save('./Data/Subject{}_{}.npy'.format(j+1,names[i-3]), subject[i])
    #for j, subject in enumerate([test_subject_1_freqs, test_subject_2_freqs]):
    #    for i in range(3,8):
    #        np.save('./Data/Test_Subject{}_{}.npy'.format(j+1,names[i-3]), subject[i])

    # Load Data
    #for i in range(3):
    #    vars()['subject_{}_freqs'.format(i+1)] = []
    #    for name in names:
    #        vars()['subject_{}_freqs'.format(i+1)].append(np.load('./Data/Subject{}_{}.npy'.format(i+1, name),\
    #                                                         allow_pickle=True))
    # test data
    #for i in range(2):
    #    vars()['test_subject_{}_freqs'.format(i+1)] = []
    #    for name in names:
    #        vars()['test_subject_{}_freqs'.format(i+1)].append(np.load('./Data/Test_Subject{}_{}.npy'.format(i+1, name),\
    #                                                         allow_pickle=True))

    # go through each band, transform it into the desired shape and extract features
    final_bands = []
    for subject in [subject_1_freqs, subject_2_freqs, subject_3_freqs]:
        for i in range(5):
            first_sample = np.append(subject[i][:31], [[0,0]], axis=0).reshape(-1,32,2)
            current_band = np.append(first_sample, subject[i][31:].reshape(21599,-1,2), axis=0)
            band_mean = np.mean(current_band, axis=1)
            band_min = np.min(current_band, axis=1)
            band_max = np.max(current_band, axis=1)
            band_energy = np.sum(np.abs(current_band)**2, axis=1)
            band_integral = np.sum(current_band, axis=1)
            
            currentBand = np.append(band_mean, \
                                    np.append(band_min,\
                                            np.append(band_max,\
                                                        np.append(band_energy, band_integral, axis=1), axis=1),\
                                                        axis=1), axis=1)
            if i==0:
                finalBand = currentBand
            else:
                finalBand = np.append(finalBand, currentBand, axis=1)
        final_bands.append(finalBand)

    emg_bands = []
    for emg in [emg_subject_1, emg_subject_2, emg_subject_3]:
        emg_max = np.max(emg, axis=1).reshape(-1,1)
        emg_energy = np.sum(np.abs(emg)**2, axis=1).reshape(-1,1)
        emg_integral = np.sum(np.abs(emg), axis=1).reshape(-1,1)
        
        emgBand = np.append(emg_max, np.append(emg_energy, emg_integral, axis=1), axis=1)
        emg_bands.append(emgBand)


    # combine eeg bands and emg bands data
    for i in range(3):
        if i==0:
            eeg_band = final_bands[i]
            emg_band = emg_bands[i]
        else:
            eeg_band = np.append(eeg_band, final_bands[i], axis=0)
            emg_band = np.append(emg_band, emg_bands[i], axis=0)
    feature_X = np.append(eeg_band, emg_band, axis=1)

    # create pandas dataframe
    columns = ['theta_ch1_mean', 'theta_ch2_mean', 'theta_ch1_min', 'theta_ch2_min',\
            'theta_ch1_max', 'theta_ch2_max', 'theta_ch1_energy', 'theta_ch2_energy', \
            'theta_ch1_integral', 'theta_ch2_integral',\
            'alow_ch1_mean', 'alow_ch2_mean', 'alow_ch1_min', 'alow_ch2_min',\
            'alow_ch1_max', 'alow_ch2_max', 'alow_ch1_energy', 'alow_ch2_energy', \
            'alow_ch1_integral', 'alow_ch2_integral',\
            'ahigh_ch1_mean', 'ahigh_ch2_mean', 'ahigh_ch1_min', 'ahigh_ch2_min',\
            'ahigh_ch1_max', 'ahigh_ch2_max', 'ahigh_ch1_energy', 'ahigh_ch2_energy', \
            'ahigh_ch1_integral', 'ahigh_ch2_integral',\
            'beta_ch1_mean', 'beta_ch2_mean', 'beta_ch1_min', 'beta_ch2_min',\
            'beta_ch1_max', 'beta_ch2_max', 'beta_ch1_energy', 'beta_ch2_energy', \
            'beta_ch1_integral', 'beta_ch2_integral',\
            'gamma_ch1_mean', 'gamma_ch2_mean', 'gamma_ch1_min', 'gamma_ch2_min',\
            'gamma_ch1_max', 'gamma_ch2_max', 'gamma_ch1_energy', 'gamma_ch2_energy', \
            'gamma_ch1_integral', 'gamma_ch2_integral',
            'emg_max', 'emg_energy', 'emg_integral']

    df = pd.DataFrame(feature_X, columns=columns)
    #df.to_csv('./Data/featureVector.csv', index=False)

    # adjust the array from the dataArray to get two days before, current day, and two days after
    features = df.to_numpy()
    # split the features into their respective subjects
    for i in range(3):
        current_features = features[i*amountOfSubjectSamples:(i+1)*amountOfSubjectSamples]    
        new_features = np.append(current_features, np.roll(current_features, 1, axis=0), axis=1)
        new_features = np.append(new_features, np.roll(current_features, 2, axis=0), axis=1)
        new_features = np.append(new_features, np.roll(current_features, -1, axis=0), axis=1)
        new_features = np.append(new_features, np.roll(current_features, -2, axis=0), axis=1)
        current_train = new_features[2:-2]
        current_y = y_train[i*amountOfSubjectSamples:(i+1)*amountOfSubjectSamples][2:-2]
        if i==0:
            X_train = current_train
            y_actual_train = current_y
        else:
            X_train = np.append(X_train, current_train, axis=0)
            y_actual_train = np.append(y_actual_train, current_y, axis=0)


    # train on the train data with leave one out cross validation, which is one entire subject
    Cs = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
    for c in Cs:
        training_X, training_y, testing_X, testing_y = splitData(X_train, y_actual_train)
        clf = make_pipeline(StandardScaler(), SVC(C=c, class_weight='balanced'))
        clf.fit(training_X, training_y)
        y_pred = clf.predict(testing_X)
        score = balanced_accuracy_score(testing_y, y_pred)
        print('C: {}, \t score: {}'.format(c, score))
    #y_pred = clf.pred(testing_X)
    #balanced_accuracy_score(testing_y, y_pred)


    training_X, training_y, testing_X, testing_y = splitData(X_train, y_actual_train)
    clf = make_pipeline(StandardScaler(), SVC(C=0.5, class_weight='balanced', probability=True))
    clf.fit(training_X, training_y)

    # TEST DATA
    # go through each band, transform it into the desired shape and extract features
    final_bands = []
    for subject in [test_subject_1_freqs, test_subject_2_freqs]:
        for i in range(5):
            first_sample = np.append(subject[i][:31], [[0,0]], axis=0).reshape(-1,32,2)
            current_band = np.append(first_sample, subject[i][31:].reshape(21599,-1,2), axis=0)
            band_mean = np.mean(current_band, axis=1)
            band_min = np.min(current_band, axis=1)
            band_max = np.max(current_band, axis=1)
            band_energy = np.sum(np.abs(current_band)**2, axis=1)
            band_integral = np.sum(current_band, axis=1)
            
            currentBand = np.append(band_mean, \
                                    np.append(band_min,\
                                            np.append(band_max,\
                                                        np.append(band_energy, band_integral, axis=1), axis=1),\
                                                        axis=1), axis=1)
            if i==0:
                finalBand = currentBand
            else:
                finalBand = np.append(finalBand, currentBand, axis=1)
        final_bands.append(finalBand)

    emg_bands = []
    for emg in [emg_test_1, emg_test_2]:
        emg_max = np.max(emg, axis=1).reshape(-1,1)
        emg_energy = np.sum(np.abs(emg)**2, axis=1).reshape(-1,1)
        emg_integral = np.sum(np.abs(emg), axis=1).reshape(-1,1)
        
        emgBand = np.append(emg_max, np.append(emg_energy, emg_integral, axis=1), axis=1)
        emg_bands.append(emgBand)
        
    # combine eeg bands and emg bands data
    for i in range(2):
        if i==0:
            eeg_band = final_bands[i]
            emg_band = emg_bands[i]
        else:
            eeg_band = np.append(eeg_band, final_bands[i], axis=0)
            emg_band = np.append(emg_band, emg_bands[i], axis=0)
    feature_X_test = np.append(eeg_band, emg_band, axis=1)

    test_df = pd.DataFrame(feature_X_test, columns=columns)
    #test_df.to_csv('./Data/featureVector_test.csv', index=False)


    # adjust the array from the dataArray to get two days before, current day, and two days after
    features = test_df.to_numpy()
    # split the features into their respective subjects
    for i in range(2):
        current_features = features[i*amountOfSubjectSamples:(i+1)*amountOfSubjectSamples]    
        new_features = np.append(current_features, np.roll(current_features, 1, axis=0), axis=1)
        new_features = np.append(new_features, np.roll(current_features, 2, axis=0), axis=1)
        new_features = np.append(new_features, np.roll(current_features, -1, axis=0), axis=1)
        new_features = np.append(new_features, np.roll(current_features, -2, axis=0), axis=1)
        current_test = new_features[2:-2]
        #current_y = y_train[i*amountOfSubjectSamples:(i+1)*amountOfSubjectSamples][2:-2]
        if i==0:
            X_test = current_test
            #y_actual_train = current_y
        else:
            X_test = np.append(X_test, current_test, axis=0)
            #y_actual_train = np.append(y_actual_train, current_y, axis=0)

    # predict on test set
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)

    # since 4 entries were deleted we need to insert them and predict them by looking at the data before/after
    halfEntries = y_pred.shape[0] // 2
    new_y = np.zeros(y_pred.shape[0] + 8)
    new_y[:2] = y_pred[0]
    new_y[2:halfEntries+2] = y_pred[:halfEntries]
    new_y[halfEntries+2:halfEntries+4] = y_pred[halfEntries-1]
    new_y[halfEntries+4:halfEntries+6] = y_pred[halfEntries]
    new_y[halfEntries+6:-2] = y_pred[halfEntries:]
    new_y[-2:] = y_pred[-1]

    # same for y_probs
    new_y_probs = np.zeros(y_probs.shape[0] + 8)
    new_y_probs[:2] = y_probs[0]
    new_y_probs[2:halfEntries+2] = y_probs[:halfEntries]
    new_y_probs[halfEntries+2:halfEntries+4] = y_probs[halfEntries-1]
    new_y_probs[halfEntries+4:halfEntries+6] = y_probs[halfEntries]
    new_y_probs[halfEntries+6:-2] = y_probs[halfEntries:]
    new_y_probs[-2:] = y_probs[-1]

    # save predictions to csv
    test_eeg = pd.read_csv('test_eeg1.csv')
    df_ids = pd.DataFrame(test_eeg['Id'])
    df_predictions = df_ids.join(pd.DataFrame(new_y, columns=['y']))
    df_predictions.to_csv('predictions.csv', index=False)
    # same for y_probs
    df_predictions_probs = df_ids.join(pd.DataFrame(new_y_probs, columns=['1','2','3']))
    df_predictions_probs.to_csv('predictions_probs.csv', index=False)