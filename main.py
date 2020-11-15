from keras.callbacks import TensorBoard, ModelCheckpoint
from cfg import IMSIZE, epoch_num, NAME, trainingRootDir, validRootDir, trainingSamplesNum, batchSize,steps_per_epoch
import model_generator as mdlGe
from data_generator import setInfo, classMapping, train_data_generator, val_data_generator

tbd = TensorBoard(log_dir=r'C:\logs\lizukaColor-places365-DeepCNN-1')
mckpt = ModelCheckpoint(
                filepath="lizukaColor-places365-DeepCNN-1-weights.best.hdf5",
                save_best_only=True,  # Only save a model if `val_loss` has improved.
                monitor="val_loss",
                verbose=1,
            )
callbacks = [tbd, mckpt]


model = mdlGe.build_model()
model._get_distribution_strategy = lambda: None

history = model.fit_generator(train_data_generator(trainingRootDir, setInfo(True), trainingSamplesNum, batchSize), epochs=epoch_num, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
model.save('./model/lizukaColor-places365-DeepCNN-1')

x_val, y1_val, y2_val = val_data_generator()
loss, deco_4_2_loss, class_glob_2_loss, deco_4_2_acc, class_glob_2_categorical_accuracy = model.evaluate(x=x_val, y=[y1_val, y2_val], batch_size=batchSize)
print(f'loss:{loss}\ndeco_4_2_loss:{deco_4_2_loss}\nclass_glob_2_loss:{class_glob_2_loss}\ndeco_4_2_acc:{deco_4_2_acc}\nclass_glob_2_categorical_accuracy:{class_glob_2_categorical_accuracy}')


