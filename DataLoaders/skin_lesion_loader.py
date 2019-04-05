import numpy as np
import h5py

from utils.augmentation_utils import brightness_augment, random_rotation, flip


class DataLoader(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.augment = cfg.data_augment
        self.data_dir = '/home/cougarnet.uh.edu/amobiny/Desktop/BNN_Skin_Lesion/data/ISIC/preprocessed_data.h5'

    def get_data(self, mode='train'):
        if mode == 'train':
            print('Loading the training data ...')
            h5f = h5py.File(self.data_dir,'r')
            self.x_train = h5f['X_train'][:]
            y_train = h5f['y_train'][:]
            h5f.close()
            self.y_train = self.one_hot(y_train)
            # self.mean = np.mean(self.x_train, axis=(0, 1, 2))
            # self.std = np.std(self.x_train, axis=(0, 1, 2))
            # self.mean = [0.485, 0.456, 0.406]
            # self.std = [0.229, 0.224, 0.225]
        elif mode == 'valid':
            print('Loading the validation data ...')
            h5f = h5py.File(self.data_dir, 'r')
            self.x_valid = h5f['X_test'][:]
            y_valid = h5f['y_test'][:]
            h5f.close()
            self.y_valid = self.one_hot(y_valid)
        elif mode == 'test':
            print('Loading the test data ...')
            h5f = h5py.File(self.data_dir, 'r')
            self.x_test = h5f['X_test'][:]
            y_test = h5f['y_test'][:]
            h5f.close()
            self.y_test = self.one_hot(y_test)

    def next_batch(self, start=None, end=None, mode='train'):
        if mode == 'train':
            x = self.x_train[start:end]
            y = self.y_train[start:end]
            # if self.augment:
                # x = flip(x)
                # x = random_rotation(x, self.cfg.max_angle)
                # x = brightness_augment(x)
        elif mode == 'valid':
            x = self.x_valid[start:end]
            y = self.y_valid[start:end]
        elif mode == 'test':
            x = self.x_test[start:end]
            y = self.y_test[start:end]
        # x = self.normalize(x)
        return x, y

    def count_num_batch(self, batch_size, mode='train'):
        if mode == 'train':
            num_batch = int(self.y_train.shape[0] / batch_size)
        elif mode == 'valid':
            num_batch = int(self.y_valid.shape[0] / batch_size)
        elif mode == 'test':
            num_batch = int(self.y_test.shape[0] / batch_size)
        return num_batch

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.y_train.shape[0])
        self.x_train = self.x_train[permutation, :, :, :]
        self.y_train = self.y_train[permutation, :]

    def one_hot(self, y):
        y_ohe = np.zeros((y.size, int(y.max() + 1)))
        y_ohe[np.arange(y.size), y.astype(int)] = 1
        return y_ohe

    def normalize(self, x):
        x /= 255.
        for channel in range(3):
            x[..., channel] = (x[..., channel] - self.mean[channel]) / self.std[channel]
        return x




