import csv
import numpy as np
from PIL import Image
import h5py

project_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/BNN_Skin_Lesion'
input_img_dir = '/data/ISIC/ISIC2018_Task3_Training_Input/'
label_dir = '/data/ISIC/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

with open(project_dir + label_dir, 'r') as f:
    reader = csv.reader(f)
    array = np.array(list(reader))

label_array = array[1:]
num_samples = label_array.shape[0]
IMG_H = 450
IMG_W = 600
X = np.zeros((num_samples, IMG_H, IMG_W, 3))
y = np.zeros(num_samples)
for i, img_name in enumerate(label_array[:, 0]):
    X[i] = np.array(Image.open(project_dir + input_img_dir + img_name + '.jpg'))
    y[i] = np.argmax(label_array[i, 1:].astype(np.float32))

h5f = h5py.File('data.h5', 'w')
h5f.create_dataset('X', data=X)
h5f.create_dataset('y', data=y)
h5f.close()




