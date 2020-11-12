import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D
from keras.models import Model
from cfg import IMSIZE


def build_model():
    input = Input(shape=(IMSIZE, IMSIZE, 1))

    layer = Conv2D(32, (1, 1), name='conv_1_1', activation='relu', padding='same', strides=(1, 1))(input)
    layer = Conv2D(32, (1, 1), name='conv_1_2',activation='relu', padding='same', strides=(1, 1))(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(64, (2, 2), name='conv_2_1', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(64, (2, 2), name='conv_2_2', activation='relu', padding='same', strides=(2, 2))(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(128, (2, 2), name='conv_3_1', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(128, (2, 2), name='conv_3_2', activation='relu', padding='same', strides=(2, 2))(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(256, (3, 3), name='conv_4_1', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(256, (3, 3), name='conv_4_2', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(256, (3, 3), name='conv_4_3', activation='relu', padding='same', strides=(2, 2))(layer)
    layer = BatchNormalization()(layer)

    layer = Conv2D(512, (3, 3), name='conv_5_1', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(512, (3, 3), name='conv_5_2', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(512, (3, 3), name='conv_5_3', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = BatchNormalization()(layer)

    layer = UpSampling2D(size=(2, 2))(layer)

    layer = Conv2D(256, (3, 3), name='conv_6_1', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(256, (3, 3), name='conv_6_2', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = BatchNormalization()(layer)

    layer = UpSampling2D(size=(2, 2))(layer)

    layer = Conv2D(128, (3, 3), name='conv_7_1', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(128, (3, 3), name='conv_7_2', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = BatchNormalization()(layer)

    layer = UpSampling2D(size=(2, 2))(layer)

    layer = Conv2D(64, (2, 2), name='conv_8_1', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(32, (2, 2), name='conv_8_2', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = Conv2D(16, (2, 2), name='conv_8_3', activation='relu', padding='same', strides=(1, 1))(layer)
    layer = BatchNormalization()(layer)

    output = Conv2D(2, (1, 1), name='conv_9_1', activation='tanh', padding='same', strides=(1, 1))(layer)

    model = Model(inputs = input, outputs = output)
    return model