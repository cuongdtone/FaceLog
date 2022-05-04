""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

import cv2
import numpy as np
import threading


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 112))
    img = (img.astype('float32') - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


def cal_distance_face(embedding_1, embedding_2):
    embedding_1 /= np.expand_dims(np.sqrt(np.sum(np.power(embedding_1, 2), 1)), 1)
    embedding_2 /= np.expand_dims(np.sqrt(np.sum(np.power(embedding_2, 2), 1)), 1)
    return np.sum(np.multiply(embedding_1, embedding_2), 1)


class CustomThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return
