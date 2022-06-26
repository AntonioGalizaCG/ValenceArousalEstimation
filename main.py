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

training = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

checkpoint_filepath = './checkpoint2'

brake = model_checkpoint_callback = ModelCheckpoint(
	filepath=checkpoint_filepath,
	save_weights_only=True,
	monitor='val_loss',
	mode='max',
	save_best_only=True)

def load_images(inputPath,n):
    counter = 0
    x=[]
    y=[]
    for image in os.listdir(inputPath+"/images/")[:n]
        try:
            annotation = float(np.load(image[:-4]+"_val.npy").tolist())
        except: 
            pass
        counter += 1







def data_split(x,p,y,mode,ratio):
    pass

def cnn(width, height, depth, filters=(32, 32, 32), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1
	# define the model input
	inputs = Input(shape=inputShape)
	# loop over the number of filters
	d=5
	x = inputs
	# CONV => RELU => BN => POOL
	x = Conv2D(filters=f, kernel_size=(d,d), strides=(1,1), padding="same")(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = MaxPooling2D(pool_size=(2, 2))(x)
	x = Dense(1, activation="linear")(x)
	model = Model(inputs, x)
	# return the CNN
	return model
# # construct the argument parser and parse the arguments
if training:
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", type=str, required=True,
		help="path to input dataset of house images")
	args = vars(ap.parse_args())

	x1,p1,y1=load_images(args["dataset"],150000)
	xt1, pt1, yt1, xv1, pv1, yv1 = data_split(x1, p1, y1, 0, .25)

	xt, pt, yt, xv, pv, yv = xt1, pt1, yt1, xv1, pv1, yv1

	xt, pt, yt, xv, pv, yv = np.array(xt), np.array(pt), np.array(yt), np.array(xv), np.array(pv), np.array(yv)

cnn = create_cnn(32, 32, 3, regress=False)

if training:
	opt = Adam(lr=1e-3, decay=1e-3/200) #RMSprop(learning_rate=0.001)
	model.compile(loss="mse", optimizer=opt)
	# #train the model
	# print("[INFO] training model...")
	cnn.fit(
		x= xt, y=yt,
		validation_data=(xv, yv),
		epochs=200, batch_size=20,shuffle=False,callbacks=[brake])
# # # make predictions on the testing data
################################################################################
if training:
	model.save_weights("model-zenbu2.h5")
else:
	#model.load_weights('model-TOP3.h5')
	model.load_weights(checkpoint_filepath)
################################################################################
#preds = model.predict([pv])
