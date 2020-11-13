import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Flatten, Dense
from keras.models import Model
from cfg import IMSIZE


def build_model():
    input = Input(shape=(IMSIZE, IMSIZE, 1))

    # global encoder
    enco_glob = Conv2D(16, (1, 1), name='enco_glob_1', activation='relu', padding='same', strides=(1, 1))(input)
    enco_glob = Conv2D(16, (1, 1), name='enco_glob_2',activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(16, (1, 1), name='enco_glob_3', activation='relu', padding='same', strides=(1, 1), dilation_rate=2)(enco_glob)
    enco_glob = BatchNormalization()(enco_glob)

    enco_glob = Conv2D(64, (1, 1), name='enco_glob_4', activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(128, (1, 1), name='enco_glob_5', activation='relu', padding='same', strides=(1, 1), dilation_rate=2)(enco_glob)
    enco_glob = BatchNormalization()(enco_glob)

    enco_glob = Conv2D(256, (3, 3), name='enco_glob_6', activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(256, (3, 3), name='enco_glob_7', activation='relu', padding='same', strides=(2, 2))(enco_glob)
    enco_glob = BatchNormalization()(enco_glob)

    enco_glob = Conv2D(512, (3, 3), name='enco_glob_8', activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(512, (3, 3), name='enco_glob_9', activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(512, (3, 3), name='enco_glob_10', activation='relu', padding='same', strides=(2, 2))(enco_glob)
    enco_glob = BatchNormalization()(enco_glob)

    enco_glob = Flatten()(enco_glob)
    enco_glob = Dense(1024, name='enco_glob_11', activation='relu')(enco_glob)
    enco_glob_class = Dense(512, name='enco_glob_12', activation='relu')(enco_glob)

    # classification
    class_glob = Dense(512, name='class_glob_1', activation='relu')(enco_glob_class)
    class_glob = Dense(434, name='class_glob_1', activation='relu')(class_glob) # 434 classes for Places MIT

    # continue global encoder
    enco_glob = Dense(512, name='enco_glob_13', activation='relu')(enco_glob_class)

    # local encoder
    enco_loco = Conv2D(64, (1, 1), name='enco_loco_1', activation='relu', padding='same', strides=(1, 1))(input)
    enco_loco = Conv2D(64, (1, 1), name='enco_loco_2', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(64, (1, 1), name='enco_loco_3', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = BatchNormalization()(enco_loco)

    enco_loco = Conv2D(128, (3, 3), name='enco_loco_4', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(128, (3, 3), name='enco_loco_5', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(256, (3, 3), name='enco_loco_6', activation='relu', padding='same', strides=(2, 2))(enco_loco)
    enco_loco = BatchNormalization()(enco_loco)

    enco_loco = Conv2D(512, (3, 3), name='enco_loco_7', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(512, (3, 3), name='enco_loco_8', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(512, (3, 3), name='enco_loco_9', activation='relu', padding='same', strides=(2, 2))(enco_loco)
    enco_loco = BatchNormalization()(enco_loco)

    # fusion
    fuse = K.repeat(enco_glob, (IMSIZE//4)*(IMSIZE//4))
    fuse = K.reshape(fuse, [IMSIZE//4, IMSIZE//4, 512])
    fuse = K.concatenate([enco_loco, fuse], axis=3)
    fuse = Conv2D(512, (1, 1), name='fusion_1', activation='relu', padding='same', strides=(1, 1))(fuse) # reduce dimension

    # decoder
    deco = Conv2D(128, (3, 3), name='deco_1_1', activation='relu', padding='same', strides=(1, 1))(fuse)
    deco = BatchNormalization()(deco)
    deco = UpSampling2D(size=(2, 2))(deco)
    deco = Conv2D(64, (3, 3), name='deco_2_1', activation='relu', padding='same', strides=(1, 1))(fuse)
    deco = Conv2D(64, (3, 3), name='deco_2_2', activation='relu', padding='same', strides=(1, 1))(fuse)
    deco = BatchNormalization()(deco)
    deco = UpSampling2D(size=(2, 2))(deco)
    deco = Conv2D(32, (3, 3), name='deco_3_1', activation='relu', padding='same', strides=(1, 1))(fuse)
    output = Conv2D(2, (3, 3), name='deco_3_2', activation='relu', padding='same', strides=(1, 1))(fuse)

    model = Model(inputs = input, outputs = output)
    return model