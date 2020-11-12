from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave, imshow
from skimage import img_as_ubyte, img_as_int
from matplotlib import pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import cv2
from cfg import IMSIZE, epoch_num, NAME
import model_generator as mdlGe

DATADIR = '.\\PetImages'
CATEGORIES = ["Dog", 'Cat']
dir = fr'.\log\{NAME}'
print(dir)
tbCall = TensorBoard(log_dir=r'C:\logs\15cats_300epo')



training_img = []

def get_train_img():
    # for ctg in CATEGORIES:
        path = './'# os.path.join(DATADIR,ctg)
        for im in os.listdir(path):
            if 'testCat' in im:
                try:
                    im_path = os.path.join(path, im).replace('\\', '/')
                    im_bgr = cv2.imread(im_path, cv2.IMREAD_COLOR)
                    im_bgr = im_bgr.astype(np.float32)
                    im_bgr /= 255.0
                    im_bgr = cv2.resize(im_bgr, (IMSIZE,IMSIZE))
                    im_lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
                    l, ab = im_lab[:,:,0], im_lab[:,:,1:]
                    ab /= 128.0
                    training_img.append([l,ab])
                except Exception as e:
                    print(f'Error {e} found at {os.path.join(path,im)}\n')


get_train_img()
X = []
y = []
random.shuffle(training_img)
for grayscale, color in training_img:
    X.append(grayscale)
    y.append(color)

X = np.array(X).reshape(-1,IMSIZE, IMSIZE, 1)
y = np.array(y).reshape(-1, IMSIZE, IMSIZE, 2)

def adam_opt():
    return Adam(lr=0.001, beta_1=0.99, beta_2=0.999)
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2, input_shape=( None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

model = mdlGe.build_model()
model.compile(optimizer='adam', loss=euclidean_distance_loss)
model._get_distribution_strategy = lambda: None

# X = cv2.imread('lena_color.tif',cv2.IMREAD_COLOR)
# X = X.astype(np.float32)
# X /= 255.0
# X = cv2.cvtColor(X, cv2.COLOR_BGR2Lab)
# # X = cv2.resize(X, (IMSIZE, IMSIZE))
# cv2.imshow('lab', X)
# L, ab = X[:,:,0], X[:,:,1:]
# ab /= 128.0
# L = L.reshape(1, L.shape[0], L.shape[1], 1)
# ab = ab.reshape(1, ab.shape[0], ab.shape[1], 2)

# image = img_to_array(load_img('lena_color.tif'))
# image = np.array(image, dtype=np.float32)
#
# X = rgb2lab(1.0/255*image)[:,:,0]
# Y = rgb2lab(1.0/255*image)[:,:,1:]
# Y /= 128
# X = X.reshape(1, 512, 512, 1)
# Y = Y.reshape(1, 512, 512, 2)

model.fit(x=X, y=y, batch_size=1, epochs=epoch_num, callbacks=[tbCall])

print(model.evaluate(X, y, batch_size=1))
model.save('./model/cat&dog_epo_5')


