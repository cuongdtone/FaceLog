from Face.face_threading import face_thread
import cv2
from queue import Queue
from threading import Thread
from Face.face_threading import *
import time

cap = cv2.VideoCapture(0)

face_threading = face_thread(cap)
final_frame_queue, frame_ori_queue = face_threading.run()
# print(final_data_queue)

width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
fps_video = fps if fps <= 120 else 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_write = cv2.VideoWriter('outputs/test.avi', fourcc, fps_video, (width, height))

fpss = []
last_time = time.time()
while True: 
    # print('task 1')

    data = final_frame_queue.get()
    frame_ori = frame_ori_queue.get()

    frame = data['frame']  # drawed frame

    fps = 1 / (time.time() - last_time)
    fpss.append(fps)
    fps = sum(fpss) / len(fpss)

    t_size = cv2.getTextSize('FPS: %d' % (fps), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, thickness=2)[0]
    cv2.rectangle(frame, (10, 10), (10 + t_size[0] + 10, 10 + t_size[1] + 10), (128, 0, 128), -1)
    cv2.putText(frame, "FPS: %d" % (fps), (10, 10 + t_size[1] + 10 - 3), cv2.FONT_HERSHEY_PLAIN,
                fontScale=2.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    video_write.write(frame)
    cv2.imshow('Video', frame)
    last_time = time.time()
    cv2.waitKey(5)
