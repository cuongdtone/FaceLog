import pickle
import numpy as np
import cv2

from utils.deepface import *
from utils.face_align import norm_crop
from glob import glob

list_pkl = glob('src/data/*.pkl')
data = {}
for path in list_pkl:
    with open(path, 'rb') as f:
        data.update({path: pickle.load(f)})


def compare_cosine(feet1, feet2):
    return np.dot(feet1, feet2) / (np.linalg.norm(feet1) * np.linalg.norm(feet2))

face_detect = RetinaFace(model_file='src/det_500m.onnx')
face_recog = ArcFaceONNX(model_file='src/w600k_mbf.onnx')

cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    faces, kpss = face_detect.detect(frame)

    for idx, kps in enumerate(kpss):
        image_size = 112
        warped = norm_crop(frame, kps)
        feet = face_recog.get_feat(warped).flatten()
        for k in data.keys():
            feet2 = data[k]
            print(k, ' : ', compare_cosine(feet, feet2))
        cv2.imshow(str(idx), warped)
    for face in faces:
        face_box = face.astype(np.int)
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 2)

    print('---------')
    cv2.imshow('Frame', frame)
    cv2.waitKey(5)

