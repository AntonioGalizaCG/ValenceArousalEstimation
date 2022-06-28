import csv
import os
import numpy as np
image_path = "/home/antonio/Downloads/AffectNet-8Labels/train_set"

with open('dataset_val.csv',"a") as file:
    file.write("id, score\n")
    for i in os.listdir(image_path+"/images/"):
        annotation = np.load(image_path+"/annotations/"+i[:-4]+"_val.npy")
        file.write(i + ", " + str(annotation)+"\n")
    file.close()

with open('dataset_aro.csv',"a") as file:
    file.write("id, score\n")
    for i in os.listdir(image_path+"/images/"):
        annotation = np.load(image_path+"/annotations/"+i[:-4]+"_aro.npy")
        file.write(i + ", " + str(annotation)+"\n")
    file.close()
