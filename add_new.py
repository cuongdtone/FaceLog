from add_new_employee import *
from utils.load_data import load_user_data
from glob import glob
import os

path_to_img_dir = '/home/cuong/Desktop/CASIA-WebFace'

list_img_dir = glob(path_to_img_dir + '/*')

user, user_info = load_user_data()

for idx, u in enumerate(user_info):
    if u['embed'] is not None: continue
    id = u['code']
    path_to_dir = list_img_dir[idx]
    feat = create_feat(path_to_dir)
    update_face_feature_employee(None, id, str(feat))
