import cv2
import numpy as np
from queue import Queue
from threading import Thread
from .face_tracker import *
from .face_detecter import RetinaFace
from .face_recognizer import ArcFaceONNX, Face
from .functions import bufferless_camera
from .functions import compute_color_for_labels
from unidecode import unidecode


class CameraDetectThread():
    def __init__(self, rtsp_url, face_detecter, face_recognizer, employees_data=None):
        self.employees_data = employees_data
        self.face_detecter = face_detecter #RetinaFace(model_file='src/det_500m.onnx')
        self.face_recognizer = face_recognizer #ArcFaceONNX(model_file='src/w600k_mbf.onnx')
        self.track = Track()
        self.ct = CentroidTracker()

        self.cam_cleaner = bufferless_camera(rtsp_url)

        self.frame_ori_queue = Queue(maxsize=2)
        self.frame_ori2_queue = Queue(maxsize=2)
        self.frame_detect_queue = Queue(maxsize=2)
        self.data_recognize_queue = Queue(maxsize=2)
        self.data_final_queue = Queue(maxsize=2)
        self.frame_final_queue = Queue(maxsize=2)
        self.period = 2
        self.count = self.period
        self.detect = Thread(target=self.detect_thread, args=[self.frame_ori_queue, self.data_recognize_queue])
        self.recognize = Thread(target=self.recognize_thread, args=[self.data_recognize_queue, self.data_final_queue])
        self.detect.setDaemon(True)
        self.recognize.setDaemon(True)

    def check_cam(self):
        return self.cam_cleaner.camera.isOpened()

    def run(self):
        self.detect.start()
        self.recognize.start()
        return self.data_final_queue, self.frame_ori_queue

    def detect_thread(self, frame_ori_queue, data_recognize_queue):
        while self.cam_cleaner.camera.isOpened():
            if self.cam_cleaner.last_frame is not None:
                frame = self.cam_cleaner.last_frame

                faces, kpss = self.face_detecter.detect(frame, input_size=(640, 640))
                put_data = {'frame': frame, 'faces': faces, 'kpss': kpss}

                data_recognize_queue.put(put_data)
                frame_ori_queue.put(frame)
        self.cam_cleaner.camera.release()

    def recognize_thread(self, data_recognize_queue, data_final_queue):
        while self.cam_cleaner.camera.isOpened():
            data = data_recognize_queue.get()
            frame = data['frame']
            faces = data['faces']
            kpss = data['kpss']
            objects, input_centroid = self.ct.update(faces)
            out_info = []
            for idx, (objectID, centroid) in enumerate(objects.items()):
                # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                face_box, kps = find_faces(objectID, objects, input_centroid, faces, kpss)
                face_box = face_box.astype(np.int)
                info = self.track.update(objectID, face_box, kps, frame, self.face_recognizer, self.employees_data, self.count >= self.period)

                text = "{}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if face_box is None:
                    continue
                color = compute_color_for_labels(objectID)
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


class MultiCameraDetectThread():
    def __init__(self, rtsp_url_list, face_detecter, face_recognizer, employees_data=None):
        self.employees_data = employees_data
        self.face_detecter = face_detecter  # RetinaFace(model_file='src/det_500m.onnx')
        self.face_recognizer = face_recognizer  # ArcFaceONNX(model_file='src/w600k_mbf.onnx')
        # self.track = Track()
        # self.ct = CentroidTracker()
        self.camera = {}
        self.track = {}
        self.ct = {}
        for idx, rtsp_url in enumerate(rtsp_url_list):
            clean_camera = bufferless_camera(rtsp_url)
            if clean_camera.check_cam():
                self.camera.update({idx: clean_camera})
                track = Track()
                ct = CentroidTracker()
                self.track.update({idx: track})
                self.ct.update({idx: ct})
        if len(self.camera.keys()) < 1:
            print("[INFO] Can't connect to all camera ! \n System exit !")
            exit(0)
        self.frame_ori_queue = Queue(maxsize=2)
        self.frame_ori2_queue = Queue(maxsize=2)
        self.frame_detect_queue = Queue(maxsize=2)
        self.data_recognize_queue = Queue(maxsize=2)
        self.data_final_queue = Queue(maxsize=2)
        self.frame_final_queue = Queue(maxsize=2)
        self.period = 2
        self.count = self.period
        self.detect = Thread(target=self.detect_thread, args=[self.frame_ori_queue, self.data_recognize_queue])
        self.recognize = Thread(target=self.recognize_thread, args=[self.data_recognize_queue, self.data_final_queue])
        self.detect.setDaemon(True)
        self.recognize.setDaemon(True)

    def check_alive_cam(self):
        for camId in self.camera.keys():
            cam = self.camera[camId]
            if cam.check_cam() is False:
                cam.release()
                del self.camera[camId]
        if len(self.camera.keys()) < 1:
            return False
        else:
            return True

    def run(self):
        self.detect.start()
        self.recognize.start()
        return self.data_final_queue, self.frame_ori_queue

    def detect_thread(self, frame_ori_queue, data_recognize_queue):
        while self.check_alive_cam():
            put_data = {}
            frames_ori = {}
            for camId in self.camera.keys():
                if self.camera[camId].last_frame is not None:
                    frame = self.camera[camId].last_frame
                    faces, kpss = self.face_detecter.detect(frame, input_size=(640, 640))
                    data = {'frame': frame, 'faces': faces, 'kpss': kpss}
                    put_data.update({camId: data})
                    frames_ori.update({camId: frame})
            data_recognize_queue.put(put_data)
            frame_ori_queue.put(frames_ori)

    def recognize_thread(self, data_recognize_queue, data_final_queue):
        while self.check_alive_cam():
            data_multi = data_recognize_queue.get()
            final_data = {}
            for camId in data_multi.keys():
                data = data_multi[camId]
                frame = data['frame']
                faces = data['faces']
                kpss = data['kpss']
                objects, input_centroid = self.ct[camId].update(faces)
                out_info = []
                for idx, (objectID, centroid) in enumerate(objects.items()):
                    # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    face_box, kps = find_faces(objectID, objects, input_centroid, faces, kpss)
                    face_box = face_box.astype(np.int)
                    info = self.track[camId].update(objectID, face_box, kps, frame, self.face_recognizer, self.employees_data,
                                             self.count >= self.period)

                    text = "{}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
                    if face_box is None:
                        continue
                    color = compute_color_for_labels(objectID)
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
                    final_data.update({camId: {'frame': frame, 'people': out_info}})

            data_final_queue.put(final_data)

            if self.count >= self.period:
                self.count = 0
            else:
                self.count += 1
        self.cap.release()

    def stop(self):
        self.cap.release()


