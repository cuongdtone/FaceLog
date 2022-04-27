import cv2
import numpy as np
from Face.face import Face_Model

face_model = Face_Model()
cap = cv2.VideoCapture('samples/trumpvsbinden.mp4') #'samples/TrumpvsBinden.mp4')
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    timer = cv2.getTickCount()
    if ret:
        faces, kpss = face_model.detect(frame)
        for idx, face in enumerate(faces):
            face_box = face.astype(np.int)
            feet = face_model.face_encoding(frame, kpss[idx])
            info = face_model.face_compare(feet)

            cv2.putText(frame, info['fullname'], (face_box[0], face_box[1]), cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 1)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        t_size = cv2.getTextSize('FPS: %d'%(fps), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, thickness=2)[0]
        cv2.rectangle(frame, (10, 10), (10+t_size[0]+10, 10+t_size[1]+10), (128, 0, 128), -1)
        cv2.putText(frame, "FPS: %d"%(fps), (10, 10+t_size[1]+10-3), cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow("Video", frame)
        cv2.waitKey(5)
    else:
        break

cap.release()