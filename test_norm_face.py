import cv2
import numpy as np
from utils.face_align import norm_crop, get_similarity_transform
from utils.deepface import RetinaFace, ArcFaceONNX


face_detect = RetinaFace(model_file='src/det_500m.onnx')
face_recog = ArcFaceONNX(model_file='src/w600k_mbf.onnx')

cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    faces, kpss = face_detect.detect(frame)

    for idx, kps in enumerate(kpss):
        image_size = 112
        M = get_similarity_transform(kps)
        warped = cv2.warpAffine(frame, M, (image_size, image_size), borderValue=0.0)
        cv2.imshow(str(idx), warped)
    for face in faces:
        face_box = face.astype(np.int)
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 2)
    cv2.imshow('Frame', frame)
    cv2.waitKey(5)
