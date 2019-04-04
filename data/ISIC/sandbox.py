import h5py
import numpy as np

h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/BNN_Skin_Lesion/data/ISIC/preprocessed_data.h5', 'r')
X_train = h5f['X_train'][:]
y_train = h5f['y_train'][:]
X_test = h5f['X_test'][:]
y_test = h5f['y_test'][:]
h5f.close()

X_train_rep = np.zeros((0, 224, 224, 3))
y_train_rep = np.array([])
for cls in range(7):
    num_rep = np.ceil(np.sum(y_train == 1) / np.sum(y_train == cls))
    X_c = X_train[y_train == cls]
    y_c = cls * np.ones(X_c.shape[0])

    X_rep = np.repeat(X_c, num_rep, axis=0)[:np.sum(y_train == 1)]
    y_rep = np.repeat(y_c, num_rep, axis=0)[:np.sum(y_train == 1)]

    X_train_rep = np.concatenate((X_train_rep, X_rep), axis=0)
    y_train_rep = np.append(y_train_rep, y_rep)

h5f = h5py.File('preprocessed_repeated_data.h5', 'w')
h5f.create_dataset('X_train', data=X_train_rep)
h5f.create_dataset('y_train', data=y_train_rep)
h5f.create_dataset('X_test', data=X_test)
h5f.create_dataset('y_test', data=y_test)
h5f.close()



print()
