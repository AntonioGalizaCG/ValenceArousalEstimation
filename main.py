# import the necessary packages
from tensorflow.keras.optimizers import Adam, RMSprop,SGD
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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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

args = vars(ap.parse_args())

training = args["training"]

if training == "1":
    training = True
else:
    training = False

net = vgg16(96,96,3) #vgg16(112, 112, 3)

global_counter = 0
global_mean = 0
def CCC(x,y):
    x_mean = K.mean(x)
    y_mean = K.mean(y)

    x_var = K.var(x)
    y_var = K.var(y)

    x_std = K.std(x)
    y_std = K.std(y)

    cov = K.mean((x - x_mean) * (y - y_mean))

    numerator = 2 * cov

    denominator = x_var**2 + y_var**2 + (x_mean - y_mean) ** 2

    return (1 - numerator / denominator) * 100

if training:
    checks = "./checkpoints/weights-{epoch:03d}-{val_loss:.4f}.hdf5"
    saver = ModelCheckpoint(checks,monitor='val_loss',verbose=1,
    save_best_only=False,save_weights_only=True,mode='min',save_freq='epoch')

    brake = EarlyStopping(
        monitor='val_loss',
        min_delta=10**-6,
        patience=3,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    train_file = "data/small_dataset.csv"
    image_dir = "/home/tony/Downloads/AffectNet-8Labels/train_set/images"
    train_label_df = pd.read_csv(train_file, delimiter=',',
                                 header=1,
                                 names=['id', 'score_val', 'score_aro'],
                                 low_memory=False)

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
                                              target_size=(96, 96),
                                              batch_size=80,
                                              subset="training")

    v_gen = train_datagen.flow_from_dataframe(dataframe=train_label_df,
                                              directory=image_dir,
                                              x_col="id",
                                              y_col=["score_val","score_aro"],
                                              class_mode="raw",
                                              target_size=(96, 96),
                                              batch_size=80,
                                              subset="validation")

    opt = Adam(learning_rate=1e-5, decay=1e-5/20) #RMSprop(learning_rate=0.0001)
    for i in range(3):print(t_gen[i])

    net.compile(loss=CCC, optimizer=opt)

    net.fit(t_gen,
            validation_data=v_gen,
            epochs=20,
            callbacks=[saver])

    net.save_weights("models/model_big.h5")

else:
    for i in os.listdir("./checkpoints"):
        if i[0]!=".":
            print("###########################################################")
            print("#   WEIGHTS:", i,)
            print("###########################################################")
            net.load_weights("./checkpoints/"+i)
            for n in range(10):
                try:
                    img=np.array([cv2.resize(cv2.imread("/home/tony/Downloads/AffectNet-8Labels/train_set/images/34050"+str(n)+".jpg"),(96,96))/255])
                    print("predicted: ", net.predict(img)[0][0], net.predict(img)[0][1])
                    print("actual: ", np.load("/home/tony/Downloads/AffectNet-8Labels/train_set/annotations/34050"+str(n)+"_val.npy"), np.load("/home/tony/Downloads/AffectNet-8Labels/train_set/annotations/34050"+str(n)+"_aro.npy"))
                except:
                    passannotation############################################################################
#preds = model.predict([pv])
