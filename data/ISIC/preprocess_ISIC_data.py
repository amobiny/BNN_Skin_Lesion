import h5py
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/BNN_Skin_Lesion/data/ISIC/data.h5', 'r')
X = h5f['X'][:]
y = h5f['y'][:]
h5f.close()

num_samples = y.shape[0]
new_img_height = 224
new_img_width = 224

print('Start re-sizing and normalizing the data ...')
X_resized = np.zeros((num_samples, new_img_height, new_img_width, 3))
for i, img in enumerate(X):
    X_resized[i] = cv2.resize(img, (new_img_width, new_img_height), interpolation=cv2.INTER_LANCZOS4) / 255.

# split train and test data
print('Start splitting the data ...')
X_train = np.zeros((0, new_img_height, new_img_width, 3))
y_train = np.array([])
X_test = np.zeros((0, new_img_height, new_img_width, 3))
y_test = np.array([])
for cls in range(7):
    print('class #{}'.format(cls))
    # get the data for each class
    X_cls = X_resized[y == cls]
    y_cls = cls * np.ones(X_cls.shape[0])
    # split and concatenate
    X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.2)
    X_train = np.concatenate((X_train, X_tr), axis=0)
    X_test = np.concatenate((X_test, X_te), axis=0)
    y_train = np.append(y_train, y_tr)
    y_test = np.append(y_test, y_te)

print('Done.')

h5f = h5py.File('preprocessed_data.h5', 'w')
h5f.create_dataset('X_train', data=X_train)
h5f.create_dataset('y_train', data=y_train)
h5f.create_dataset('X_test', data=X_test)
h5f.create_dataset('y_test', data=y_test)
h5f.close()
