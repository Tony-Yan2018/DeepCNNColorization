import time
import cv2
import os
import natsort
import tensorflow as tf
from colorize_pic import batchGeneratorFps, processPrediction

model = tf.keras.models.load_model('./model/lizukaColor-places365-DeepCNN-3')


def videoToFrames(path):
    videoStream = cv2.VideoCapture(path)
    i = 0
    fps = videoStream.get(cv2.CAP_PROP_FPS)
    prev = 0
    while (videoStream.isOpened()):
        time_elapsed = time.time() - prev
        ret, frame = videoStream.read()
        if ret == True:
            cv2.imwrite('./video/frames/' + str(i) + '.jpg', frame)
            i += 1
            if time_elapsed > 1. / fps:
                prev = time.time()
        else:
            break
    videoStream.release()
    cv2.destroyAllWindows()
    return round(fps)


def colorFrames(bwPath, fps):
    for framesPerSec in batchGeneratorFps(bwPath, fps):
        for each in framesPerSec:
            output = model.predict([each])
            processPrediction([each], output)
            time.sleep(1) # keep frame order


def framesToVideo(path, videoName, fps):
    filesList = natsort.natsorted(os.listdir(path), reverse=False)
    out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280,720))
    for filename in filesList:
        print(path + filename)
        img = cv2.imread(path + filename)
        out.write(img)
    out.release()


fps = videoToFrames('./video/Yellow Sky Traile.mp4')
colorFrames('./video/frames/',  fps)
framesToVideo('./video/color/','./video/Yellow Sky Traile_colored.mp4',fps)