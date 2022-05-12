import time
from utils.face import FaceRecog
import cv2
import os
from utils.utils_calc import save_new_image
from utils.sqlite_database import update_pkl_employee
import argparse

id = input('ID : ')
camera = 2

# Phase 1: Select ROI and take img

db_root = 'src/data'
person = os.path.join(db_root, id)
try:
    os.mkdir(person)
except:
    pass

Cap = True
if Cap:
    cap = cv2.VideoCapture(camera)
    ret, frame = cap.read()
    r = cv2.selectROI(frame)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.rectangle(frame,
                      (int(r[0])-2, int(r[1])-2),
                      (int(r[0]+r[2])+2, int(r[1]+r[3])+2),
                      color=(0, 0, 255),
                      thickness=2)
        if not ret: break
        cv2.imshow('ROI selector', frame)
        key = cv2.waitKey(5)
        if key == ord(' '):
            save_new_image(person, frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])])
            time.sleep(0.3)
        elif key == ord('q'):
            break
    cap.release()

# Phase 2: create pkl file
face_model = FaceRecog(None)
face_model.create_data_file(db_root, id)

# Phase 3: update DB
update_pkl_employee(None, id, id + '.pkl')

# Phase 4: delete img dir
