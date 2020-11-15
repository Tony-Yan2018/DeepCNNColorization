import cv2
import numpy as np
from cfg import IMSIZE
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave, imshow
from skimage import img_as_ubyte, img_as_int

def bgr2lab(im_path):
    '''openCV method to transform an image from bgr space to lab space.

    :param im_path: image path to read
    :return: a tuple of Lab, first L(0~100), then ab(-1~+1)
    '''
    im_bgr = cv2.imread(im_path, cv2.IMREAD_COLOR)
    im_bgr = im_bgr.astype(np.float32)
    im_bgr /= 255.0
    im_bgr = cv2.resize(im_bgr, (IMSIZE, IMSIZE))
    im_lab = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LAB)
    l, ab = im_lab[:, :, 0], im_lab[:, :, 1:]
    ab /= 128.0
    return l, ab

def rgb2lab(im_path):
    '''skimage method to transform an image from bgr space to lab space.

    :param im_path: image path to read
    :return: a tuple of Lab, first X(0~100), then Y(-1~+1)
    '''
    image = img_to_array(load_img(im_path))
    image = np.array(image, dtype=np.float32)

    X = rgb2lab(1.0/255*image)[:,:,0]
    Y = rgb2lab(1.0/255*image)[:,:,1:]
    Y /= 128
    X = X.reshape(1, 512, 512, 1)
    Y = Y.reshape(1, 512, 512, 2)

    return X,Y
