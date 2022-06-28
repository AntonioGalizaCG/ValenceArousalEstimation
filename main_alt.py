# import the necessary packages
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
# from pyimagesearch import datasets
# from pyimagesearch import models
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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add, concatenate, average
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.data import Dataset
from ast import literal_eval
from math import pi
import tensorflow

from keras.preprocessing.image import ImageDataGenerator

physical_devices = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

training = 1

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

checkpoint_filepath = './checkpoint_arousal_big'

brake = model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_loss',
	mode='min',
	save_best_only=True)

train_file = "dataset_aro.csv"
image_dir = "/home/antonio/Downloads/AffectNet-8Labels/train_set/images"

train_label_df = pd.read_csv(train_file, delimiter=',', header=1, names=['id', 'score'],low_memory=False)

print (train_label_df.dtypes)

train_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip = True,
                                   fill_mode = "nearest", zoom_range = 0.2,
                                   width_shift_range = 0.2, height_shift_range=0.2,
                                   rotation_range=30,validation_split=0.25)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_label_df, directory=image_dir,
                                              x_col="id", y_col="score",
                                              class_mode="raw", target_size=(112, 112),
                                              batch_size=10,)



# def load_images(inputPath,n):
# 	counter = 0
# 	x=[]
# 	y=[]
# 	for image in os.listdir(inputPath+"/images/")[:n]:
# 		try:
# 			annotation = float(np.load(inputPath+"/annotations/"+image[:-4]+"_aro.npy").tolist())
# 			cv_img = cv2.resize(cv2.imread(inputPath+"/images/"+image),(112,112))/255
# 			x.append(cv_img)
# 			y.append((annotation+1)/2)
# 			counter += 1
# 			#print(counter/n*100,"%")
# 		except Exception as e:
# 			print(e)
# 	return x, y
#
#
# def data_split(x, y, ratio, mode="random"):
# 	xy = [(x[i],y[i]) for i in range(len(x))]
# 	counter = 0
# 	length = len(xy)
# 	val_x = list()
# 	val_y = list()
#
# 	if mode == "random":
# 		from random import shuffle
# 		shuffle(xy)
# 	while counter/length < ratio:
# 		chosen = xy.pop(counter)
# 		val_x.append(chosen[0])
# 		val_y.append(chosen[1])
# 		counter += 1
#
# 	train_x = [i[0] for i in xy]
# 	train_y = [i[1] for i in xy]
#
# 	return np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y)


def create_cnn(width, height, depth, filters=(32, 32, 32), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	# loop over the number of filters
	d=5
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		if i == 0:
			x = inputs
		# CONV => RELU => BN => POOL
		x = Conv2D(filters=f, kernel_size=(d,d), strides=(1,1), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)
		d-=1
	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(48, activation="relu")(x)
	x = Dense(24, activation="relu")(x)
	x = Dense(1, activation="sigmoid")(x)
	model = Model(inputs, x)
	# return the CNN
	return model
# # construct the argument parser and parse the arguments
cnn = create_cnn(112, 112, 3, regress=False)

if training:
	# my_x, my_y = load_images("/home/antonio/Downloads/AffectNet-8Labels/train_set",10000)
	# t_x, t_y, v_x, v_y = data_split(my_x, my_y, .25)
	# del my_x
	# del my_y

	opt = Adam(learning_rate=1e-3, decay=1e-3/200) #RMSprop(learning_rate=0.001)
	cnn.compile(loss="mse", optimizer=opt)
	# #train the model
	# print("[INFO] training model...")
	# cnn.fit(
	# x= t_x, y=t_y,
	# validation_data=(v_x, v_y),
	# epochs=20, batch_size=100,shuffle=False,callbacks=[brake])
	cnn.fit(train_generator,epochs=20)
	cnn.save_weights("model-arousal_big.h5")
else:
	#model.load_weights('model-TOP3.h5')
	cnn.load_weights("model-arousal_big.h5")
	img=np.array([cv2.resize(cv2.imread("/home/antonio/Downloads/AffectNet-8Labels/train_set/images/340600.jpg"),(112,112))/255])
	print((cnn.predict(img)[0][0]*2)-1)
	print(np.load("/home/antonio/Downloads/AffectNet-8Labels/train_set/annotations/340600_aro.npy"))

################################################################################
#preds = model.predict([pv])
