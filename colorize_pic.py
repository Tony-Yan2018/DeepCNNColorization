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
from model_generator import euclidean_distance_loss, loss_class, losses

toColorize = []
output_ab = []

def prepTestData():
    for file in os.listdir(testRootDir):
        L, ab = bgr2lab(testRootDir+'/'+file)
        L = np.array(L).reshape(1, L.shape[0], L.shape[1], 1)
        toColorize.append(L)
    return toColorize

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

singleImPath = './val_256/Places365_val_00000164.jpg'


model = tf.keras.models.load_model('./model/lizukaColor-places365-DeepCNN-3')#, custom_objects={'euclidean_distance_loss': euclidean_distance_loss,'loss_class':loss_class})

output_ab = model.predict(prepSingleImg(singleImPath))

def processPrediction(input, pred):
    if len(input) == len(pred):
        for i in range(len(pred)):
            image = lab2bgr(input[i], pred[i])
            cv2.imshow('BGR',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # imsave('./result/image_'+str(i)+'.jpg', image)

processPrediction(prepSingleImg(singleImPath), output_ab)

