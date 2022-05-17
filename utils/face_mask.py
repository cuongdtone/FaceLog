""" Created by MrBBS """
# 11/9/2021
# -*-encoding:utf-8-*-

from tensorflow.keras.models import load_model
import cv2
import numpy as np

mask_detection = load_model('./src/mask_detection.h5')


def MaskDetect(image):
    try:
        face = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (16, 16))
        face = np.expand_dims(np.expand_dims(face, axis=2), axis=0)
        label = int(list(mask_detection.predict(face)[0]).index(1.))
        if label == 1:
            return True
    except:
        pass
    return False
