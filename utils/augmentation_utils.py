import random
import scipy
import numpy as np
import cv2
import tensorflow as tf


def tf_aug(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
    # img = tf.image.random_contrast(img)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    # img = tf.image.crop_and_resize(img)
    return img


def random_rotation(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).
    Arguments:
    max_angle: `float`. The maximum rotation angle.
    Returns:
    batch of rotated 2D images
    """
    size = batch.shape
    batch = np.squeeze(batch)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            image = np.squeeze(batch[i])
            angle = random.uniform(-max_angle, max_angle)
            batch_rot[i] = scipy.ndimage.interpolation.rotate(image, angle, mode='nearest', reshape=False)
        else:
            batch_rot[i] = batch[i]
    return batch_rot.reshape(size)


def brightness_augment(batch, factor=0.5):
    size = batch.shape
    batch = np.squeeze(batch)
    batch_aug = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        if bool(random.getrandbits(1)):
            img = np.squeeze(batch[i])
            hsv = cv2.cvtColor(img*255, cv2.COLOR_RGB2HSV)  # convert to hsv
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform())    # scale channel V uniformly
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255  # reset out of range values
            batch_aug[i] = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
        else:
            batch_aug[i] = batch[i]
    return batch_aug.reshape(size)


def flip(batch):
    size = batch.shape
    batch = np.squeeze(batch)
    batch_aug = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        img = np.squeeze(batch[i])
        if bool(random.getrandbits(1)):     # left/ right flip
            img = np.fliplr(img)
        if bool(random.getrandbits(1)):  # up/ down flip
            img = np.flipud(img)
        batch_aug[i] = img
    return batch_aug.reshape(size)


def add_noise(batch, mean=0, var=0.1, amount=0.01, mode='pepper'):
    original_size = batch.shape
    batch = np.squeeze(batch)
    batch_noisy = np.zeros(batch.shape)
    for ii in range(batch.shape[0]):
        image = np.squeeze(batch[ii])
        if mode == 'gaussian':
            gauss = np.random.normal(mean, var, image.shape)
            image = image + gauss
        elif mode == 'pepper':
            num_pepper = np.ceil(amount * image.size)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        elif mode == "s&p":
            s_vs_p = 0.5
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            image[coords] = 1
            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            image[coords] = 0
        batch_noisy[ii] = image
    return batch_noisy.reshape(original_size)

