import os
import numpy as np
import pandas as pd
import glob
from PIL import Image

train_dir = os.listdir('C:/Users/ziang/Desktop/COMPSYS_302_Project_1/data/car_data/train/')
train_csv = pd.read_csv('C:/Users/ziang/Desktop/COMPSYS_302_Project_1/data/stanford-car-dataset-by-classes-folder/anno_train.csv', header=None)
data_labels = np.array(pd.read_csv('C:/Users/ziang/Desktop/COMPSYS_302_Project_1/data/stanford-car-dataset-by-classes-folder/names.csv', header=None))


os.chdir('C:/Users/ziang/Desktop/COMPSYS_302_Project_1/data/car_data/train/')
for folder in train_dir:
    folder_read = folder + '/*.jpg'

    # reading from each class directory folder
    filelist = glob.glob(folder_read)
    all_files = [fname[-9:] for fname in filelist]
    sc_data = []
    for i in range(len(filelist)):
        sc_data.append(np.array(Image.open(filelist[i])))
    # sc_data = np.array([Image.open(fname) for fname in filelist])
    for i in range(len(sc_data)):
        file_dir = folder + '/' + all_files[i]
        annot = train_csv.loc[train_csv[0] == all_files[i]]
        x1 = annot[1][annot.index[0]]
        y1 = annot[2][annot.index[0]]
        x2 = annot[3][annot.index[0]]
        y2 = annot[4][annot.index[0]]
        Image.fromarray(sc_data[i][y1:y2, x1:x2]).save(file_dir)
        print("cropped" + file_dir)
