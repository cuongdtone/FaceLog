# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 2022-04-27
# @Function      : Device workflow

import os
import time
from threading import Thread
import cv2
import yaml
from utils.log import checkin
from utils.face_threading import FaceThread as FaceThread
from test_new_thead_FAS import FaceThread2
from utils.load_data import load_user_data


class Main:
    def __init__(self):
        self.face_ids = {}
        self.flag_reload = False
        self.config = yaml.load(open("src/settings.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        self.employees = []  # danh sách nhân viên
        self.employees_data = []  # code_name_faceid
        self.load()
        # Front camera thread
        self.front_camera = cv2.VideoCapture(2)
        self.front_camera_thread = FaceThread2(self.front_camera, self.employees_data)
        self.checkin_list = {}
        # Back camera thread
        # self.back_camera = cv2.VideoCapture(0)
        # self.back_camera_thread = FaceThread(self.back_camera, self.employees_data)
        # self.checkout_list = {}
        # Reload database
        reload_thread = Thread(target=self.reload_local_db)
        # reload_thread.start()

    def load(self):
        self.employees_data, self.employees = load_user_data()

    def reload_local_db(self):
        while True:
            self.load()
            self.front_camera_thread.face_model.employees_data = self.employees_data  # sync data to face recognition model
            time.sleep(3)

    def run(self):
        final_data_front_queue, frame_ori_front_queue = self.front_camera_thread.run()
        # final_data_back_queue, frame_ori_back_queue = self.back_camera_thread.run()
        while self.front_camera.isOpened(): # or self.back_camera.isOpened():
            if self.front_camera.isOpened():
                #front camera
                data_front = final_data_front_queue.get()
                frame_ori_front = frame_ori_front_queue.get()
                frame_front = data_front['frame']  # drawed frame
                people_front = data_front['people']
                self.checkin_list = checkin(people_front, self.checkin_list, self.employees, frame_ori_front, self.config)
                cv2.imshow('Front Camera', frame_front)
            # if self.back_camera.isOpened():
            #     data_back = final_data_back_queue.get()
            #     frame_ori_back = frame_ori_back_queue.get()
            #     frame_back = data_back['frame']  # drawed frame
            #     people_back = data_back['people']
            #     cv2.imshow('Back Camera', frame_back)
            cv2.waitKey(5)


if __name__ == '__main__':
    Main().run()
