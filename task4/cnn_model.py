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

model.summary()