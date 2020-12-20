c.f. SPINDLE: End-to-end learning from EEG/EMG to extrapolate animal sleep scoring across experimental settings, labs and species (D. Miladinovic, et al.)\
Preprocessing steps:\
```python preprocessing_train_eeg.py``` (labels included) \
```python preprocessing_emg.py``` \
```python preprocessing_test_eeg.py``` 
(preprocessing.py as a whole. Separated to avoid memory explosion.)

CNN architecture design:\
```python cnn_model.py```

CNN train: (modify the training params as you wish)\
```python cnn_sub12.py```\
```python cnn_sub23.py```\
```python cnn_sub13.py```

CNN predict on test set:\
```python cnn_predict.py```

Result write to files:\
```python result_generate.py```
