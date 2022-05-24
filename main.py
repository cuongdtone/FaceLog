# -*-encoding:utf-8-*-

import os
import cv2
import yaml
from utils.face_detecter import RetinaFace
from utils.face_recognizer import ArcFaceONNX, Face
from utils.face_landmark import TDDFA_ONNX
from utils.face_mask import MaskDetection
from utils.load_data import load_user_data
from utils.timekeeping import checkin # check_timekeep
from utils.functions import Watcher
from utils.face_thread import MultiCameraDetectThread


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
        Watcher('src/TimeKeepingDB.db', self.load) # reload db while changes

    def load(self):
        self.employees_data, self.search_tree, self.employees_info = load_user_data()

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
