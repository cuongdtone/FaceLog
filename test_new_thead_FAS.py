import cv2
from utils.load_data import load_user_data
from threading import Thread
from queue import Queue
import numpy as np
from utils.face import FaceRecog
from utils.face_tracker import CentroidTracker, find_faces, Follow_track
from utils.load_data import load_user_data
from unidecode import unidecode




palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

class CameraBufferCleanerThread(Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()

class FaceThread2():
    def __init__(self, cap, employees_data=None):
        self.face_model = FaceRecog(employees_data)
        self.track = Follow_track()
        self.ct = CentroidTracker()
        self.cap = cap
        self.cam_cleaner = CameraBufferCleanerThread(self.cap)

        self.frame_ori_queue = Queue(maxsize=2)
        self.frame_ori2_queue = Queue(maxsize=2)
        self.frame_detect_queue = Queue(maxsize=2)
        self.data_recognize_queue = Queue(maxsize=2)
        self.data_final_queue = Queue(maxsize=2)
        self.frame_final_queue = Queue(maxsize=2)
        self.period = 5
        self.count = self.period
        self.detect = Thread(target=self.detect_thread, args=[self.frame_ori_queue, self.data_recognize_queue])
        self.recognize = Thread(target=self.recognize_thread, args=[self.data_recognize_queue, self.data_final_queue])
        self.detect.setDaemon(True)
        self.recognize.setDaemon(True)

    def run(self):
        self.detect.start()
        self.recognize.start()
        return self.data_final_queue, self.frame_ori_queue

    def detect_thread(self, frame_ori_queue, data_recognize_queue):
        while self.cap.isOpened():
            if self.cam_cleaner.last_frame is not None:
                frame = self.cam_cleaner.last_frame
                faces, kpss = self.face_model.detect(frame)
                put_data = {'frame': frame, 'faces': faces, 'kpss': kpss}
                data_recognize_queue.put(put_data)
                frame_ori_queue.put(frame)
        self.cap.release()

    def recognize_thread(self, data_recognize_queue, data_final_queue):
        # global count, period
        while self.cap.isOpened():
            data = data_recognize_queue.get()
            frame = data['frame']
            faces = data['faces']
            kpss = data['kpss']
            objects, input_centroid = self.ct.update(faces)
            out_info = []
            for idx, (objectID, centroid) in enumerate(objects.items()):
                # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                face_box, kps = find_faces(objectID, objects, input_centroid, faces, kpss)
                info = self.track.update(objectID, face_box, kps, frame, self.face_model, self.count >= self.period)

                text = "{}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if face_box is None:
                    continue
                color = compute_color_for_labels(objectID*10)
                cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, 2)
                if info is None:
                    text = 'Verifing'
                else:
                    text = unidecode(info['fullname'].split()[-1] + ' - %.2f' % (info['Sim']))
                    out_info.append(info)
                t_size = cv2.getTextSize(text,
                                         fontFace=cv2.FONT_HERSHEY_PLAIN,
                                         fontScale=1.0, thickness=1)[0]
                cv2.putText(frame, text, (int(face_box[0]), int(face_box[1] + t_size[1] + 5)), cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1.0, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            if self.count >= self.period:
                self.count = 0
            else:
                self.count += 1
            data_final_queue.put({'frame': frame, 'people': out_info})
        self.cap.release()
    def stop(self):
        self.cap.release()

if __name__ == '__main__':
    cap = cv2.VideoCapture(2)
    employees_data, employees = load_user_data()
    thread = FaceThread2(cap, employees_data)
    data, frame_queue = thread.run()

    while cap.isOpened():
        frame = frame_queue.get()
        frame = data.get()['frame']
        cv2.imshow('cc', frame)
        cv2.waitKey(5)

