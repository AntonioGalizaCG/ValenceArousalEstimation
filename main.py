# import the necessary packages
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split

import numpy as np
import argparse
import locale
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import glob
import cv2

import tensorflow

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (BatchNormalization,
  									 Conv2D,
									 MaxPooling2D,
									 Activation,
									 Dropout,
									 Dense,
									 Flatten,
									 Input,
									 add,
									 concatenate,
									 average)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.data import Dataset


from keras.preprocessing.image import ImageDataGenerator

from vgg16 import vgg16

# physical_devices = tensorflow.config.list_physical_devices('GPU')
# tensorflow.config.experimental.set_memory_growth(physical_devices[0],
#                                                  enable=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", type=str, required=True,
	help="tells if training is enabled.")
ap.add_argument("-m", "--mode", type=str, required=True,
	help="mode is valence or arousal.")
args = vars(ap.parse_args())

training = bool(args["training"])
mode = args["mode"]

extension = ""
if mode == "valence":
	extension = "val"
else:
	extension = "aro"

net = vgg16(112, 112, 3)

if training:
	brake = EarlyStopping(
	    monitor='val_loss',
	    min_delta=0.001,
	    patience=3,
	    verbose=1,
	    mode='auto',
	    baseline=None,
	    restore_best_weights=True
	)
	train_file = "data/dataset_"+extension+".csv"
	image_dir = "/home/tony/Downloads/AffectNet-8Labels/train_set/images"
	train_label_df = pd.read_csv(train_file, delimiter=',',
	                             header=1,
								 names=['id', 'score'],
								 low_memory=False)

	print (train_label_df.dtypes)

	train_datagen = ImageDataGenerator(rescale = 1./255,
									   horizontal_flip = True,
	                                   fill_mode = "nearest",
									   zoom_range = 0.2,
	                                   width_shift_range = 0.2,
									   height_shift_range=0.2,
	                                   rotation_range=30,
									   validation_split=0.25)

	t_gen = train_datagen.flow_from_dataframe(dataframe=train_label_df,
	                                          directory=image_dir,
	                                          x_col="id",
											  y_col="score",
	                                          class_mode="raw",
											  target_size=(112, 112),
	                                          batch_size=10,
											  subset="training")

	v_gen = train_datagen.flow_from_dataframe(dataframe=train_label_df,
	                                          directory=image_dir,
	                                          x_col="id",
											  y_col="score",
	                                          class_mode="raw",
											  target_size=(112, 112),
	                                          batch_size=10,
											  subset="validation")

	opt = Adam(learning_rate=1e-3, decay=1e-3/10) #RMSprop(learning_rate=0.001)

	net.compile(loss="mse", optimizer=opt)

	net.fit(t_gen,
	        validation_data=v_gen,
			epochs=10,
			callbacks=[brake])

	net.save_weights("models/model-"+extension+"_big.h5")

else:
	net.load_weights("models/model-"+extension+"_big.h5")
	img=np.array([cv2.resize(cv2.imread("/home/antonio/Downloads/AffectNet-8Labels/train_set/images/340600.jpg"),(112,112))/255])
	print(net.predict(img)[0][0]*2-1)
	print(np.load("/home/antonio/Downloads/AffectNet-8Labels/train_set/annotations/340600_"+extension+".npy"))

################################################################################
#preds = model.predict([pv])
