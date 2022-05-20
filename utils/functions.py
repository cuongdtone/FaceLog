""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

import threading
import multiprocessing as mp
import cv2
import os
import os.path as osp
from pathlib import Path
import pickle
import time


def get_object(name):
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj



class bufferless_camera(threading.Thread):
    def __init__(self, rtsp_url, name='camera-buffer-cleaner-thread'):
        self.rtsp = rtsp_url
        self.camera = cv2.VideoCapture(rtsp_url)
        self.fps = 25
        self.last_frame = None
        super(bufferless_camera, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()
            # if last_frame is not None:
            #     self.last_frame = last_frame
            # else:
            #     print('[INFO] Reconnecting !')
            #     self.reconnect()

    def reconnect(self):
        self.camera.release()
        del self.camera
        time.sleep(0.01)
        self.camera = cv2.VideoCapture(self.rtsp)

    def get_frame(self):

        self.last_time = time.time()
        return self.last_frame


    def check_cam(self):
        return self.camera.isOpened()

    def get_resolution(self):
        width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return (height, width)


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

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)