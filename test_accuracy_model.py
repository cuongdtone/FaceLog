from glob import glob
from Face.face import Face_Model
import cv2
import numpy as np


face_model = Face_Model()

list_img = glob('/home/cuong/Desktop/face_dataset/*/*')

# for img in list_img:
#     try:
#         image = cv2.imread(img)
#
#         reuslt = face.detect(image)[0][0]
#         box = [int(i) for i in reuslt]
#         face_img = image[box[1]:box[3], box[0]:box[2], :]
#         cv2.imwrite(img, face_img)
#         print(reuslt)
#     except:
#         pass
for img in list_img:
    print(img)
    image = cv2.imread(img)

    faces, kpss = face_model.detect(image)
    print(faces)
    for idx, face in enumerate(faces):
        face_box = face.astype(np.int)
        feet = face_model.face_encoding(image, kpss[idx])
    