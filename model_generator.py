import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Flatten, Dense, RepeatVector, Reshape
from keras.layers.merge import concatenate
from keras.models import Model
from cfg import IMSIZE, MName
from keras.utils import plot_model
from keras.losses import categorical_crossentropy

def euclidean_distance_loss(y_true, y_pred):
    """
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def loss_class(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)

losses = {'deco_4_2': euclidean_distance_loss,
          'class_glob_2': loss_class,
          }

my_metrics ={
    'deco_4_2':'acc',
    'class_glob_2':'categorical_accuracy',
    }


def build_model():
    input = Input(shape=(IMSIZE, IMSIZE, 1,))

    # global encoder
    enco_glob = Conv2D(16, (1, 1), name='enco_glob_1', activation='relu', padding='same', strides=(1, 1))(input)
    enco_glob = Conv2D(16, (1, 1), name='enco_glob_2',activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(16, (1, 1), name='enco_glob_3', activation='relu', padding='same', strides=(2, 2))(enco_glob)
    enco_glob = BatchNormalization()(enco_glob)

    enco_glob = Conv2D(64, (3, 3), name='enco_glob_4', activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(128, (3, 3), name='enco_glob_5', activation='relu', padding='same', strides=(2, 2))(enco_glob)
    enco_glob = BatchNormalization()(enco_glob)

    enco_glob = Conv2D(256, (3, 3), name='enco_glob_6', activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(256, (3, 3), name='enco_glob_7', activation='relu', padding='same', strides=(2, 2))(enco_glob)
    enco_glob = BatchNormalization()(enco_glob)

    enco_glob = Conv2D(512, (3, 3), name='enco_glob_8', activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(512, (3, 3), name='enco_glob_9', activation='relu', padding='same', strides=(1, 1))(enco_glob)
    enco_glob = Conv2D(512, (3, 3), name='enco_glob_10', activation='relu', padding='same', strides=(2, 2))(enco_glob)
    enco_glob = BatchNormalization()(enco_glob)

    enco_glob = Flatten()(enco_glob)
    enco_glob = Dense(768, name='enco_glob_11', activation='relu')(enco_glob)
    enco_glob_class = Dense(512, name='enco_glob_12', activation='relu')(enco_glob)

    # classification
    class_glob = Dense(512, name='class_glob_1', activation='relu')(enco_glob_class)
    class_glob = Dense(365, name='class_glob_2', activation='softmax')(class_glob) # 365 classes for Places MIT

    # continue global encoder
    enco_glob = Dense(512, name='enco_glob_13', activation='relu')(enco_glob_class)

    # local encoder
    enco_loco = Conv2D(64, (1, 1), name='enco_loco_1', activation='relu', padding='same', strides=(1, 1))(input)
    enco_loco = Conv2D(64, (1, 1), name='enco_loco_2', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(64, (1, 1), name='enco_loco_3', activation='relu', padding='same', strides=(2, 2))(enco_loco)
    enco_loco = BatchNormalization()(enco_loco)

    enco_loco = Conv2D(128, (3, 3), name='enco_loco_4', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(128, (3, 3), name='enco_loco_5', activation='relu', padding='same', strides=(2, 2))(enco_loco)
    enco_loco = Conv2D(256, (3, 3), name='enco_loco_6', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(256, (3, 3), name='enco_loco_7', activation='relu', padding='same', strides=(2, 2))(enco_loco)
    enco_loco = BatchNormalization()(enco_loco)

    enco_loco = Conv2D(512, (3, 3), name='enco_loco_8', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(512, (3, 3), name='enco_loco_9', activation='relu', padding='same', strides=(1, 1))(enco_loco)
    enco_loco = Conv2D(512, (3, 3), name='enco_loco_10', activation='relu', padding='same', strides=(2, 2))(enco_loco)
    enco_loco = BatchNormalization()(enco_loco)

    # fusion
    print(f'enco_loco:{enco_loco.shape}') #enco_loco:(None, 16, 16, 512)
    print(f'enco_glob:{enco_glob.shape}')#enco_glob:(None, 512)
    reshaped_shape = enco_loco.shape[:3].concatenate(enco_glob.shape[1])#reshaped_shape:(None, 16, 16, 512)
    print(f'reshaped_shape:{reshaped_shape}')
    print(f'enco_glob.shape:{enco_glob.shape}')#enco_glob.shape:(None, 512)
    fuse = RepeatVector(enco_loco.shape[1]*enco_loco.shape[2])(enco_glob)#fuse.shape:(None, 256, 512)
    print(f'fuse.shape:{fuse.shape}')
    fuse = Reshape(( reshaped_shape[1], reshaped_shape[2], reshaped_shape[3]))(fuse)
    print(f'fuse.shape:{fuse.shape}')
    fuse = concatenate([enco_loco, fuse], axis=3)
    print(f'fuse.shape:{fuse.shape}')
    fuse = Conv2D(512, (1, 1), name='fusion_1', activation='relu', padding='same', strides=(1, 1))(fuse) # reduce dimension
    fuse = UpSampling2D(size=(2, 2))(fuse)
    # decoder
    deco = Conv2D(128, (3, 3), name='deco_1_1', activation='relu', padding='same', strides=(1, 1))(fuse)
    deco = Conv2D(128, (3, 3), name='deco_1_2', activation='relu', padding='same', strides=(1, 1))(deco)
    deco = BatchNormalization()(deco)

    deco = UpSampling2D(size=(2, 2))(deco)
    deco = Conv2D(64, (3, 3), name='deco_2_1', activation='relu', padding='same', strides=(1, 1))(deco)
    deco = Conv2D(64, (3, 3), name='deco_2_2', activation='relu', padding='same', strides=(1, 1))(deco)
    deco = BatchNormalization()(deco)

    deco = UpSampling2D(size=(2, 2))(deco)
    deco = Conv2D(32, (3, 3), name='deco_3_1', activation='relu', padding='same', strides=(1, 1))(deco)
    deco = Conv2D(32, (3, 3), name='deco_3_2', activation='relu', padding='same', strides=(1, 1))(deco)
    deco = BatchNormalization()(deco)

    deco = UpSampling2D(size=(2, 2))(deco)
    output = Conv2D(2, (3, 3), name='deco_4_2', activation='tanh', padding='same', strides=(1, 1))(deco)

    model = Model(inputs = input, outputs = [output, class_glob], name=MName)
    model.compile(optimizer='adam', loss=losses, metrics=my_metrics)

    print(model.metrics_names)
    # summarize model
    print(model.summary())
    plot_model(model, to_file=f'Model_Summary_{MName}.png', show_shapes=True)
    return model
