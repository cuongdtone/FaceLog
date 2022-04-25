import cv2
import numpy as np
from .deepface import Face, RetinaFace, ArcFaceONNX
import time
import os
import glob
import pickle

class Face_Model():
    def __init__(self, root_path='Db'):
        # task is "half create":  create from directory image of employee
        # task is "full create":  create with take photos
        # task is "load":  load infor to use
        self.root_path = root_path
        self.Face_Recognition = ArcFaceONNX(model_file='Face/w600k_mbf.onnx')
        self.Face_Detection = RetinaFace(model_file='Face/det_500m.onnx')
        # if task == 'half create': 
        #     self.create_data_file()
        # elif task == 'full create': 
        #     #mkdir dir
        #     #take photo
        #     #create data file
        #     pass
        # elif task == 'load': 
        #     #load data file:  position, office, feets
        #     pass
    def detect(self, img):
        return self.Face_Detection.detect(img, max_num=0, metric='default', input_size=(640, 640))
    def create_data_file(self, name, position, office):
        path_to_dir = os.path.join(self.root_path, name)
        # position = input("Position:  ")
        # office = input("Office:  ")
        list_img = glob.glob(path_to_dir+ '/*.jpg') + \
                   glob.glob(path_to_dir+ '/*.jpeg') + \
                   glob.glob(path_to_dir+ '/*.png')
        feets = {}
        for i in list_img: 
            image = cv2.imread(i)
            #convert all format image to jpg
            if i.split('.')[-1] != 'jpg': 
                os.remove(i)
                path = i.split('.')
                path.pop()
                cv2.imwrite('.'.join(path) + '.jpg', image)
                
            id_img = i.split('/')[-1].split('.')[0]
            try: 
                faces, kpss = self.Face_Detection.detect(image, max_num=0, metric='default', input_size=(640, 640))
                feet = self.face_encoding(image, kpss[0])
                feet = {id_img: feet}
                feets.update(feet)
            except: 
                continue
        data = {'Name':  name, "Position":  position, "Office": office, 'feets': feets}
        with open(path_to_dir + '/data.pkl', 'wb') as f:
            pickle.dump(data, f)
    def load_data(self, name): 
        path = os.path.join(self.root_path, name) + '/data.pkl'
        if os.path.exists(path): 
            with open(path, 'rb') as f: 
                data = pickle.load(f)
            return data
        else: 
            return False
    def face_encoding(self, image, kps): 
        face_box_class = {'kps':  kps}
        face_box_class = Face(face_box_class)
        feet = self.Face_Recognition.get(image, face_box_class)
        return feet
    def face_compare(self, feet, threshold=0.3):
        name_list = glob.glob(self.root_path+'/*')
        max_sim = -1
        info = {'Name': 'uknown', 'Sim':  max_sim, 'Position': 'None', "Office":  'None', 'path': 'icon/unknown_person.jpg'}
        for i in name_list: 
            if os.path.exists(i+'/data.pkl'):  # need load db to RAM before at ver2
                with open(i+'/data.pkl', 'rb') as f: 
                    data = pickle.load(f)
                feets = data['feets']
                for key in feets.keys(): 
                    feet_compare = feets[key]
                    sim = self.Face_Recognition.compute_sim(feet, feet_compare)
                    if sim>threshold and sim>max_sim: 
                        max_sim = sim
                        info['Name'] = data["Name"]
                        info['Sim'] = sim
                        info['Position'] = data['Position']
                        info["Office"] = data['Office']
                        info['path'] = i + '/' + key + '.jpg'
            else: 
                continue
        return info

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = Face_Model()
    a.create_data_file('Trump')
    a.create_data_file('Binden')
    # data = a.load_data('Binden')
    # print(data)

    image = cv2.imread('samples/test2.jpg')
    faces, kpss = a.detect(image)

    time_last = time.time()

    feet = a.face_encoding(image, kpss[0])
    info = a.face_compare(feet)
    time_compare = time.time() - time_last
    print("Time per one compare : ", time_compare)
    print("FPS of compare phase:  ", 1/time_compare)
    print(info)

    plt.subplot(1, 2, 1)
    plt.imshow(image[..., ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.imread(info['path'])[..., ::-1])
    plt.show()






