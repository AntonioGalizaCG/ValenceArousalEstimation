# import the necessary packages
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adadelta
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
import getpass

import tensorflow

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.layers import (BatchNormalization,
                                       Conv2D,
                                     MaxPooling2D,
                                     Activation,
                                     Dropout,
                                     Dense,
                                     Input,
                                     add,
                                     concatenate,
                                     Flatten,
                                     average)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.data import Dataset

from keras.preprocessing.image import ImageDataGenerator

from models import * #, create_cnn
import matplotlib.pyplot as plt

# physical_devices = tensorflow.config.list_physical_devices('GPU')
# tensorflow.config.experimental.set_memory_growth(physical_devices[0],
#                                                  enable=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--training", type=str, required=True,
    help="tells if training is enabled.")

ap.add_argument("-m", "--mode", type=str, required=True,
    help="selects valence or arousal network training/testing.")

args = vars(ap.parse_args())

training = args["training"]


if training == "1":
    training = True

else:
    training = False

sufix = ""

if args["mode"] == "valence":
    sufix = "val"
else:
    sufix = "aro"

net = resnet_18() #vgg16(112, 112, 3)

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

    denominator = x_var + y_var + (y_mean-x_mean)**2

    return 100*(1 - (numerator/denominator))

if training:
    checks = "./checkpoints/weights-{epoch:03d}-{val_loss:.4f}.hdf5"
    saver = ModelCheckpoint(checks,monitor='val_loss',verbose=1,
    save_best_only=False,save_weights_only=True,mode='min',save_freq='epoch')

    brake = EarlyStopping(
        monitor='val_loss',
        min_delta=10**-6,
        patience=500,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    train_file = "data/small_dataset.csv"
    image_dir = "/home/"+str(getpass.getuser())+"/Downloads/AffectNet-8Labels/train_set/images"
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
                                              y_col="score_" + sufix, #,"score_aro"],
                                              class_mode="raw",
                                              target_size=(96, 96),
                                              batch_size=80,
                                              subset="training")

    v_gen = train_datagen.flow_from_dataframe(dataframe=train_label_df,
                                              directory=image_dir,
                                              x_col="id",
                                              y_col="score_" + sufix,#,"score_aro"],
                                              class_mode="raw",
                                              target_size=(96, 96),
                                              batch_size=80,
                                              subset="validation")

    opt = Adam(learning_rate=1e-4, decay=1e-4/200) #RMSprop(learning_rate=0.0001)
    for i in range(3):print(t_gen[i])

    net.compile(loss=CCC, optimizer=opt)

    net.fit(t_gen,
            validation_data=v_gen,
            epochs=200,
            callbacks=[brake])

    net.save_weights("models/model_arousal.h5")

else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tensorflow.get_logger().setLevel('ERROR')
    image_path = "/home/"+str(getpass.getuser())+"/Downloads/AffectNet-8Labels/train_set/images/"
    labels_path = "/home/"+str(getpass.getuser())+"/Downloads/AffectNet-8Labels/train_set/annotations/"
    # for i in os.listdir("./checkpoints"):
    #     if i[0]!=".":
    i = "model_arousal.h5"
    net.load_weights("./models/"+i)
    #average_error_val = 0
    average_error = 0
    #act_val_hist = []
    act_hist = []
    #pred_val_hist = []
    pred_hist = []
    count = 0
    for image in os.listdir(image_path)[:100000]:
        try:
            print(image_path+image)
            img = np.array([cv2.resize(cv2.imread(image_path+image),(96,96))/255])
            pred =  float(net.predict(img)[0])
            if abs(pred_aro)>1: pred_aro/=abs(pred_aro)
            actual = float(np.load(labels_path + image[:-4] + "_" + sufix + ".npy"))
            abs_error = abs(actual-pred)
            act_hist.append(actual)
            pred_hist.append(pred)
            average_error += abs_error
            count += 1
        except Exception as e:
            print(e)
            pass

    plt.suptitle("Valence_Comparison")
    plt.plot(act_aro_hist, label="act aro")
    plt.plot(pred_aro_hist, label="pred aro")
    #axs[1].plot(act_aro_hist, label="act aro")
    #axs[1].plot(pred_aro_hist, label="pred aro")
    leg = plt.legend(loc='upper center')
    plt.scatter(list(range(len(act_hist))), act_hist)
    plt.scatter(list(range(len(act_hist))), pred_hist)
    plt.grid(True)
    plt.text(0.3, 0.8, "Average error: " + str(int(1000*average_error/count)/1000), size=15, color='purple')
    plt.savefig("Comp_weights_" +sufix+ str(i)+".png")


# else:
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     tensorflow.get_logger().setLevel('ERROR')
#     image_path = "/home/"+str(getpass.getuser())+"/Downloads/AffectNet-8Labels/val_set/images/"
#     labels_path = "/home/"+str(getpass.getuser())+"/Downloads/AffectNet-8Labels/val_set/annotations/"
#     # for i in os.listdir("./checkpoints"):
#     #     if i[0]!=".":
#     i = "model_valence.h5"
#     net.load_weights("./models/"+i)
#     average_error_val = 0
#     #average_error_aro = 0
#     act_val_hist = []
#     #act_aro_hist = []
#     pred_val_hist = []
#     #pred_aro_hist = []
#     for image in os.listdir(image_path)[:10000]:
#         try:
#             print(image_path+image)
#             img = np.array([cv2.resize(cv2.imread(image_path+image),(96,96))/255])
#             pred_val = float(net.predict(img)[0])
#             #pred_aro =  float(net.predict(img)[0][1])
#             actual_val = float(np.load(labels_path + image[:-4] + "_val.npy"))
#             #actual_aro = float(np.load(labels_path + image[:-4] + "_aro.npy"))
#             error_val = abs(actual_val-pred_val)#/actual_val
#             #error_aro = abs(actual_aro-pred_aro)#/actual_aro
#             act_val_hist.append(actual_val)
#             #act_aro_hist.append(actual_aro)
#             pred_val_hist.append(pred_val)
#             #pred_aro_hist.append(pred_aro)
#             average_error_val += error_val
#             #average_error_aro += error_aro
#             # print("predicted: ", pred_val, pred_aro)
#             # print("actual: ",actual_val, actual_aro)
#             # print("error: ", error_val, error_aro)
#             # print("------------------------------------------------------")
#
#         except Exception as e:
#             print(e)
#             pass
    #
    # plt.suptitle("Valence_Comparison")
    # plt.plot(act_val_hist, label="act val")
    # plt.plot(pred_val_hist, label="pred val")
    # #axs[1].plot(act_aro_hist, label="act aro")
    # #axs[1].plot(pred_aro_hist, label="pred aro")
    # plt.text(0.3, 0.8, "Average error: " + str(int(1000*average_error_val/10000)/1000), size=15, color='purple')
    # leg = plt.legend(loc='upper center')
    # plt.scatter(list(range(len(act_val_hist))), act_val_hist)
    # plt.scatter(list(range(len(act_val_hist))), pred_val_hist)
    # plt.grid(True)
    # plt.savefig("Val_Comp_weights_" + str(i)+".png")

                    ############################################################################
#preds = model.predict([pv])
