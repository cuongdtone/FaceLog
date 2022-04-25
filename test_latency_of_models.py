import cv2
import numpy as np
from Face.deepface import RetinaFace
from Face.deepface import ArcFaceONNX
from Face.deepface import Face
import time
import os

time_start_load = time.time()
total_memory, used_memory_before, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
Face_recognition = ArcFaceONNX(model_file='Face/w600k_mbf.onnx')
Face_model = RetinaFace(model_file='Face/det_500m.onnx')
total_memory, used_memory_after, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
time_end_load = time.time()
print("Used memmory for load model: ", used_memory_after - used_memory_before, "MB")
print("Time need to load model: %.3f second"%(time_end_load - time_start_load))


image = cv2.imread('samples/binden.jpg')

list_time_need_detect = []
list_time_need_get_feet = []
for i in range(10):
    time_start_detect = time.time()
    faces, kpss = Face_model.detect(image, max_num=0, metric='default', input_size=(640, 640))
    face_box_class = {'kps': kpss[0]}
    face_box_class = Face(face_box_class)
    time_need_detect = time.time() - time_start_detect

    time_start_get_feet = time.time()
    feet1 = Face_recognition.get(image, face_box_class)
    time_need_get_feet = time.time() - time_start_get_feet

    list_time_need_detect.append(time_need_detect)
    list_time_need_get_feet.append(time_need_get_feet)


mean_time_need_detect = np.array(list_time_need_detect).mean()
mean_time_need_get_feet = np.array(list_time_need_get_feet).mean()

print("Time to detection: %.3f sencond"%(mean_time_need_detect))
print("Time to get feet: %.3f sencond"%(mean_time_need_get_feet))
print("FPS of detect: ", 1/mean_time_need_detect)
print("FPS of get feet: ", 1/mean_time_need_get_feet)
print("Using multi thread, period of get feet/detect = %0.3f frame"%(mean_time_need_get_feet/mean_time_need_detect))
print("Shape of feet: ", feet1.shape)