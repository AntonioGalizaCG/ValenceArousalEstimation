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

from tensorflow.keras import backend as K

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

from vgg16 import vgg16#, create_cnn

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

training = args["training"]

if training == "1":
    training = True
else:
    training = False

mode = args["mode"]

extension = ""
if mode == "valence":
	extension = "val"
else:
	extension = "aro"

net = vgg16(112,112,3) #vgg16(112, 112, 3)
net.summary()


def CCC(x,y):
    print(x.shape, y.shape)
    x_mean = K.mean(x)
    y_mean = K.mean(y)

    x_var = K.var(x)
    y_var = K.var(y)

    x_std = K.std(x)
    y_std = K.std(y)

    cov = K.mean((x - x_mean) * (y - y_mean))

    numerator = 2 * cov * x_std * y_std

    denominator = x_var**2 + y_var**2 + (x_mean - y_mean) ** 2

    return (1 - numerator / denominator) * 100

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
	train_file = "data/dataset.csv"
	image_dir = "/home/tony/Downloads/AffectNet-8Labels/train_set/images"
	train_label_df = pd.read_csv(train_file, delimiter=',',
	                             header=1,
								 names=['id', 'score_val', 'score_aro'],
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
											  y_col=["score_val","score_aro"],
	                                          class_mode="raw",
											  target_size=(112, 112),
	                                          batch_size=80,
											  subset="training")

	v_gen = train_datagen.flow_from_dataframe(dataframe=train_label_df,
	                                          directory=image_dir,
	                                          x_col="id",
											  y_col=["score_val","score_aro"],
	                                          class_mode="raw",
											  target_size=(112, 112),
	                                          batch_size=80,
											  subset="validation")

	opt = Adam(learning_rate=1e-3, decay=1e-3/10) #RMSprop(learning_rate=0.001)
	for i in range(3):print(t_gen[i])

	net.compile(loss=CCC, optimizer=opt)

	net.fit(t_gen,
	        validation_data=v_gen,
			epochs=10,
			callbacks=[brake])

	net.save_weights("models/model-"+extension+"_big.h5")

else:
	net.load_weights("models/model-"+extension+"_big.h5")
	img=np.array([cv2.resize(cv2.imread("/home/tony/Downloads/AffectNet-8Labels/train_set/images/340501.jpg"),(112,112))/255])
	print("predicted: ", net.predict(img)[0][0]*2-1)
	print("actual: ", np.load("/home/tony/Downloads/AffectNet-8Labels/train_set/annotations/340501_"+extension+".npy"))

################################################################################
#preds = model.predict([pv])
