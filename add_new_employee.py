from utils.face import FaceRecog
import cv2
import os
from utils.utils import save_new_image

cap = cv2.VideoCapture(0)

# Phase 1: take image
name = input('Name: ')
id = input('ID code: ')

db_root = 'src/data'
person = os.path.join(db_root, name)
try:
    os.mkdir(person)
except:
    pass

Cap = True
if Cap:
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('Capture', frame)
        key = cv2.waitKey(5)
        if key == ord(' '):
            save_new_image(person, frame)
        elif key == ord('q'):
            break
    cap.release()


face_model = FaceRecog()
face_model.create_data_file(db_root, name, id)

# delete img dir then
