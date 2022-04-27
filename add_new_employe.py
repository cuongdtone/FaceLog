from Face.face import Face_Model
import cv2
import os
from Face.utils import save_new_image

cap = cv2.VideoCapture(0)

# Phase 1: take image
name = input('Name: ')
id = input('ID code: ')

db_root = 'data'
person = os.path.join(db_root, name)
try:
    os.mkdir(person)
except:
    pass

Cap = False
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


face_model = Face_Model()
face_model.create_data_file(db_root, name, id)

# delete img dir then
