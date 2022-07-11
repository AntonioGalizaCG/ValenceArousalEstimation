import csv
import os
import numpy as np
image_path = "/home/tony/Downloads/AffectNet-8Labels/train_set"

# with open('dataset_val.csv',"a") as file:
#     file.write("id, score\n")
#     for i in os.listdir(image_path+"/images/"):
#         annotation = (float(np.load(image_path+"/annotations/"+i[:-4]+"_val.npy").tolist())+1)/2
#         file.write(i + ", " + str(annotation)+"\n")
#     file.close()

with open('data/dataset.csv',"a") as file:
    file.write("id, score_val, score_aro\n")
    for i in os.listdir(image_path+"/images/"):
        annotation_val = (float(np.load(image_path+"/annotations/"+i[:-4]+"_val.npy").tolist())+1)/2
        annotation_aro = (float(np.load(image_path+"/annotations/"+i[:-4]+"_aro.npy").tolist())+1)/2
        file.write(i+", "+str(annotation_val)+", " +str(annotation_aro)+"\n")
    file.close()
