import time
import numpy as np

IMSIZE = 256
epoch_num = 10
batchSize = 16
trainingSamplesNum = 36600
steps_per_epoch = np.floor(trainingSamplesNum/batchSize)
NAME = f'Colorization-{IMSIZE}*{IMSIZE}-{epoch_num}epochs-L2Norm-15cats-{int(time.time())}'
MName = 'lizukaColor-places365-DeepCNN-1' # model name
# picNum_EachClass = 100
trainingRootDir = './data_256'
validRootDir = './val_256'
