import cv2
import numpy as np
from utils.face_align import norm_crop, get_similarity_transform
from utils.deepface import RetinaFace, ArcFaceONNX

import torch

face_detect = RetinaFace(model_file='src/det_500m.onnx')
face_recog = ArcFaceONNX(model_file='src/w600k_mbf.onnx')

cap = cv2.VideoCapture(2)

feat2 = np.ones((512))
while cap.isOpened():
    ret, frame = cap.read()
    faces, kpss = face_detect.detect(frame)

    for idx, kps in enumerate(kpss):
        image_size = 112
        M = get_similarity_transform(kps)
        warped = cv2.warpAffine(frame, M, (image_size, image_size), borderValue=0.0)
        feat = face_recog.get_feat(warped).flatten()
        sim = np.dot(feat, feat2) / (np.linalg.norm(feat) * np.linalg.norm(feat2))

        print(sim)


        cv2.imshow(str(idx), warped)
    for idx, face in enumerate(faces):
        face_box = face.astype(np.int)
        kps = kpss[idx]
        for kp in kps:
            cv2.circle(frame, (kp[0], kp[1]), color=(0, 0, 255), thickness=1, radius=2)
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    cv2.waitKey(5)
