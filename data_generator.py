import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from cfg import IMSIZE
import numpy as np
# from numpy import random
import random
import pandas as pd
import os
from image_Process import bgr2lab

'''Places365-Standard has 1,803,460 training images
with the image number per class varying from 3,068
to 5,000. The validation set has 50 images per class
and the test set has 900 images per class.'''

# data augmenter
data_aug = ImageDataGenerator(rotation_range=20,
                              shear_range=0.15,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              fill_mode="nearest")

# randIndex = random.sample(range(1803459), 36600)
# print(randIndex)

def setInfo(select): # turn the path of each image to its corresponding class
    '''

    :param select: True: training set; False: validation set
    :return: info: dist with imPath as keys; infoT: tuple converted from info
    '''
    info = dict()
    if select == True: # train set
        path = './filelist_places365-standard/places365_train_standard.txt'
    else : # valid set
        path = './filelist_places365-standard/places365_val.txt'
    with open(path) as train_file:
        for line in train_file:
            im_path = line.strip().split(' ')[0]
            im_class = line.strip().split(' ')[1]
            info[im_path] = im_class
    infoT = [(k, v) for k, v in info.items()] # unchangeable
    return info, infoT


def classMapping():
    '''

    :return: classMap: integer class number to class name;
                oneHotMap: integer class number to one-hot vector
    '''
    classMap = {}
    with open('./filelist_places365-standard/categories_places365.txt') as class_file:
        for line in class_file:
            class_name = line.strip().split(' ')[0][3:]
            class_num = line.strip().split(' ')[1]
            classMap[class_num] = class_name
    oneHotKeys = list(classMap.keys())
    oneHotVals = to_categorical(oneHotKeys, 365)
    oneHotMap = {oneHotKeys[i]: oneHotVals[i] for i in range(len(oneHotKeys))}
    return classMap, oneHotMap

def train_data_generator(trainingRootDir, trainingSetInfo, traningSampleNum, batchSize):
    randIndex = random.sample(range(1803459), traningSampleNum)
    # get unique traningSampleNum random samples from 1803459 pics
    bs=0
    while 1:
        for idx in randIndex:
            if bs == 0:
                L_train, ab_train, class_train = list(), list(), list()
            imPath = trainingRootDir + trainingSetInfo[1][idx][0]
            imClass = trainingSetInfo[0][trainingSetInfo[1][idx][0]]
            L, ab = bgr2lab(imPath)
            oneHot = classMapping()[1][imClass]
            L_train.append(L)
            ab_train.append(ab)
            class_train.append(oneHot)

            bs += 1
            if bs==batchSize: # end of current batch
                bs=0
                yield [np.array(L_train)[:,:,:,np.newaxis], np.array(ab_train)] # , np.array(class_train)]]

def val_data_generator(validRootDir ):
    L_valid, ab_valid, class_valid = list(), list(), list()
    for eachPic in os.listdir(validRootDir):
        imPath = validRootDir + '/' + eachPic
        L, ab = bgr2lab(imPath)
        oneHotLabel = classMapping()[1][setInfo(False)[0][eachPic]]
        L_valid.append(L)
        ab_valid.append(ab)
        class_valid.append(oneHotLabel)
    return np.array(L_valid), np.array(ab_valid), np.array(class_valid)

