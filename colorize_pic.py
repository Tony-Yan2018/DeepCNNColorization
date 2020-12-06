import cv2
import tensorflow as tf
import os
import numpy as np
import keras.backend as K
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from cfg import IMSIZE, testRootDir
from image_Process import bgr2lab, lab2bgr
from ISR.models import RRDN
import natsort
import time
from model_generator import euclidean_distance_loss, loss_class, losses

toColorize = []
output_ab = []

rrdn = RRDN(weights='gans')

def prepTestData():
    for file in os.listdir(testRootDir):
        L, ab = bgr2lab(testRootDir+'/'+file)
        L = np.array(L).reshape(1, L.shape[0], L.shape[1], 1)
        toColorize.append(L)
    return toColorize

def batchGeneratorFps(path, fps):
    imArr = []
    filesList = natsort.natsorted(os.listdir(path), reverse=False)
    current = 0
    for file in filesList:
        L, ab = bgr2lab(path+file)
        L = np.array(L).reshape(1, L.shape[0], L.shape[1], 1)
        imArr.append(L)
        current += 1
        if current == fps:
            current = 0
            imArr = np.array(imArr)
            yield imArr
            imArr=[]

def prepSingleImg(imPath):
    '''

    :param imPath:
    :return: the L layer of a single image
    '''
    im, ab = bgr2lab(imPath)
    imH, imW = im.shape
    im = np.array(im).reshape(1, IMSIZE, IMSIZE, 1)
    return [im]

# prepTestData()

singleImPath = './val_256/Places365_val_00000174.jpg'
# im = cv2.imread(singleImPath, cv2.IMREAD_COLOR)
# # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# im = cv2.resize(im, (320, 320), cv2.INTER_CUBIC)
# # im = np.array(im)
# # rdn = RDN(weights='psnr-large')
# sr_im = rrdn.predict(im)
#
# cv2.imshow('RGB', sr_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# model = tf.keras.models.load_model('./model/lizukaColor-places365-DeepCNN-3')#, custom_objects={'euclidean_distance_loss': euclidean_distance_loss,'loss_class':loss_class})

# output_ab = model.predict(prepSingleImg(singleImPath))

# def batchPred():


def processPrediction(input, pred):
    if len(input) == len(pred):
        for i in range(len(pred)):
            image = lab2bgr(input[i], pred[i])
            image *= 255
            image = image.astype(np.uint8)
            image = cv2.resize(image, (1280, 720), cv2.INTER_CUBIC)

            # image = rdn.predict(image)
            cv2.imwrite('./video/color/'+str(time.time())+'.jpg',image)

            # cv2.imshow('BGR',image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

# processPrediction(prepSingleImg(singleImPath), output_ab)

