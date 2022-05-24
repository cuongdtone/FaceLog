""" Created by MrBBS """
# 11/1/2021
# -*-encoding:utf-8-*-

import os
# import winsound # playsound
import threading
import time

import cv2
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import base64
from io import BytesIO
# import asyncio
from collections import Counter
import yaml
from pathlib import Path
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX, Face
from utils.face_landmark import TDDFA_ONNX
from utils.face_mask import MaskDetection

from utils.load_data import load_user_data
from utils.timekeeping import checkin # check_timekeep
from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer
from typing import Callable
from utils.face_thread import MultiCameraDetectThread


class File_Events_Handler(PatternMatchingEventHandler):
    def __init__(self, callback: Callable):
        PatternMatchingEventHandler.__init__(self, patterns=['*.[jp][pn]*'], ignore_directories=True)
        self.callback = callback

    def on_any_event(self, event):
        print(event.event_type, event.src_path)
        self.callback(isReload=True)


class Main:
    def __init__(self):
        self.config = yaml.load(open("src/settings.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        self.load()
        self.face_detecter = RetinaFace(model_file='src/det_500m.onnx')
        self.face_recognizer = ArcFaceONNX(model_file='src/w600k_mbf.onnx')
        self.tddfa = TDDFA_ONNX(**(yaml.load(open('src/mb1_120x120.yml'), Loader=yaml.SafeLoader)))
        self.face_mask = MaskDetection(weight='src/model_mask.onnx')

        # Multi (1, 2, 3 ...) camera detecter backend
        camera_list = [self.config['id_camera_1'], self.config['id_camera_2']]
        self.list_camera_name = ['front', 'back']
        self.camera_detect = MultiCameraDetectThread(camera_list,
                                                     self.face_detecter,
                                                     self.face_recognizer,
                                                     self.tddfa,
                                                     self.face_mask,
                                                     self.employees_data,
                                                     self.search_tree)
        self.checkin_list = {}
        self.checkout_list = {}

        reload = threading.Thread(target=self.reload, args=[])
        reload.start()
        # print(self.employees_data)
        # observer = Observer()
        # handler = File_Events_Handler(self.load)
        # observer.schedule(handler, self.config['data_path'], recursive=True)
        # observer.start()

    def load(self):
        self.employees_data, self.search_tree, self.employees_info = load_user_data()

    def reload(self):
        while True:
            time.sleep(3)
            self.load()
            self.camera_detect.employees_data = self.employees_data
            self.camera_detect.search_tree = self.search_tree

    def run(self):
        data_queue, frame_queue = self.camera_detect.run()
        while self.camera_detect.check_alive_cam():
            data = data_queue.get()
            frames = frame_queue.get()
            for camId in data.keys():
                cv2.imshow(self.list_camera_name[camId], data[camId]['frame'])
                if True: #self.list_camera_name[camId] == 'front':
                    self.checkin_list = checkin(data[camId]['people'],
                                                self.checkin_list,
                                                self.employees_info,
                                                frames[camId],
                                                self.config)
            cv2.waitKey(2)


if __name__ == '__main__':
    Main().run()
