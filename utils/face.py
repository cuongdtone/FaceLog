import cv2
import numpy as np
from .deepface import Face, RetinaFace, ArcFaceONNX
import time
import os
import glob
import pickle
import json
from .load_data import load_user_data

class FaceRecog():
    def __init__(self, employess_data):
        self.face_recognition = ArcFaceONNX(model_file='src/w600k_mbf.onnx')
        self.face_detection = RetinaFace(model_file='src/det_500m.onnx')
        self.employees_data = employess_data

    def create_data_file(self, db_path, name, id_code):
        path_to_dir = os.path.join(db_path, name)
        # position = input("Position:  ")
        # office = input("Office:  ")
        list_img = glob.glob(path_to_dir + '/*.jpg') + \
                   glob.glob(path_to_dir + '/*.jpeg') + \
                   glob.glob(path_to_dir + '/*.png')
        feets = []
        for i in list_img:
            image = cv2.imread(i)
            # convert all format image to jpg
            if i.split('.')[-1] != 'jpg':
                os.remove(i)
                path = i.split('.')
                path.pop()
                cv2.imwrite('.'.join(path) + '.jpg', image)

            id_img = i.split('/')[-1].split('.')[0]
            try:
                faces, kpss = self.face_detection.detect(image, max_num=0, metric='default', input_size=(640, 640))
                feet = self.face_encoding(image, kpss[0])
                feets.append(feet)
            except:
                continue
        data = {'feets': feets}
        with open(db_path + '/' + str(id_code) + '.pkl', 'wb') as f:
            pickle.dump(data, f)

    def detect(self, img):
        return self.face_detection.detect(img, max_num=0, metric='default', input_size=(640, 640))

    def face_encoding(self, image, kps): 
        face_box_class = {'kps':  kps}
        face_box_class = Face(face_box_class)
        feet = self.face_recognition.get(image, face_box_class)
        return feet

    def face_compare(self, feet, threshold=0.6):
        max_sim = -1
        info = {'fullname': 'uknown', 'Sim':  max_sim, 'code': None}
        for data in self.employees_data:
            feets = data['feets']
            for feet_compare in feets:
                sim = self.face_recognition.compute_sim(feet, feet_compare)
                info['Sim'] = sim
                if sim>threshold and sim>max_sim:
                    max_sim = sim
                    info['fullname'] = data["fullname"]
                    info['Sim'] = sim
                    info['code'] = data['code']
        return info







