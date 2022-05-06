import cv2
import numpy as np
from .face import FaceRecog
from queue import Queue
from threading import Thread
from .utils import compute_color_for_labels, get_center, center_match
from unidecode import unidecode


class FaceThread():
    def __init__(self, cap, employees_data):
        self.face_model = FaceRecog(employees_data)
        self.cap = cap
        self.frame_ori_queue = Queue(maxsize=2)
        self.frame_ori2_queue = Queue(maxsize=2)
        self.frame_detect_queue = Queue(maxsize=2)
        self.data_recognize_queue = Queue(maxsize=2)
        self.data_final_queue = Queue(maxsize=2)
        self.frame_final_queue = Queue(maxsize=2)
        self.data_output_queue = Queue(maxsize=2)
        self.period = 1
        self.count = 1
        self.out_people = [{'fullname': 'uknown', 'code': None, 'Sim': 0, 'center': np.array([0, 0])}]
        self.read = Thread(target=self.read_thread, args=[self.cap, self.frame_ori_queue, self.frame_detect_queue])
        self.detect = Thread(target=self.detect_thread, args=[self.cap, self.face_model, self.frame_detect_queue, self.data_recognize_queue])
        self.recognize = Thread(target=self.recognize_thread, args=[self.cap, self.face_model, self.data_recognize_queue, self.data_final_queue])
        self.draw = Thread(target=self.draw_thread, args=[self.cap, self.frame_ori_queue, self.data_final_queue, self.frame_final_queue, self.frame_ori2_queue])

    def run(self):
        self.read.start()
        self.detect.start()
        self.recognize.start()
        self.draw.start()
        return self.frame_final_queue, self.frame_ori2_queue

    def read_thread(self, cap, frame_ori_queue, frame_detect_queue):
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, flipCode=1)
            if not ret:
                break
            frame_ori_queue.put(frame)
            frame_detect_queue.put(frame)
        cap.release()

    def detect_thread(self, cap, face_model, frame_detect_queue, data_recognize_queue):
        while cap.isOpened():
            frame = frame_detect_queue.get()
            self.count += 1
            faces, kpss = face_model.detect(frame)
            put_data = {'frame': frame, 'faces': faces, 'kpss': kpss}
            data_recognize_queue.put(put_data)
        cap.release()

    def recognize_thread(self, cap, face_model, data_recognize_queue, data_final_queue):
        # global count, period
        while cap.isOpened():
            data = data_recognize_queue.get()
            frame = data['frame']
            faces = data['faces']
            kpss = data['kpss']
            people = []
            if self.count >= self.period:
                self.count = 0
                for idx, kps in enumerate(kpss):
                    feet = face_model.face_encoding(frame, kps)
                    info = face_model.face_compare(feet)
                    info.update({'box': faces[idx]})
                    people.append(info)
            final_data = {'faces': faces, 'people': people}
            data_final_queue.put(final_data)
        cap.release()

    def draw_thread(self, cap, frame_ori_queue, data_final_queue, frame_final_queue, frame_ori2_queue):
        global out_people
        while cap.isOpened():
            frame = frame_ori_queue.get()
            frame_ori2_queue.put(frame.copy())
            data = data_final_queue.get()
            faces = data['faces']
            people = data['people']
            for idx, face in enumerate(faces):
                face_box = face.astype(np.int)
                if people == []:
                    people = self.out_people
                self.out_people = people
                try:
                    now_center = get_center(face_box)
                    i = center_match(now_center, people)
                    info = people[i]
                    color = compute_color_for_labels(sum([ord(character) for character in info['fullname']]))
                    name_display = unidecode(info['fullname'].split()[-1] + ' - %.2f' % (info['Sim']))
                    t_size = cv2.getTextSize(name_display,
                                             fontFace=cv2.FONT_HERSHEY_PLAIN,
                                             fontScale=1.0, thickness=1)[0]
                    cv2.rectangle(frame,
                                  (face_box[0], face_box[1]),
                                  (face_box[0] + t_size[0] + 10, face_box[1] + t_size[1] + 10),
                                  color, -1)
                    cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), color, 2)
                    cv2.putText(frame, name_display, (face_box[0], face_box[1] + t_size[1] + 5), cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1.0, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
                except Exception as e:
                    continue
            out_info = []
            for person in people:
                info = {'fullname': person['fullname'], 'code': person['code']}
                out_info.append(info)
            frame_final_queue.put({'frame': frame, 'people': out_info})
        cap.release()

    def stop(self):
        self.cap.release()


if __name__ == '__main__':
    cap = cv2.VideoCapture('samples/binden.mp4')
    face_recog = FaceThread(cap)
    frame_final_queue = face_recog.run()
    while True:
        timer = cv2.getTickCount()
        frame = frame_final_queue.get()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        t_size = cv2.getTextSize('FPS: %d'%(fps), cv2.FONT_HERSHEY_PLAIN, fontScale=2.0, thickness=2)[0]
        cv2.rectangle(frame, (10, 10), (10+t_size[0]+10, 10+t_size[1]+10), (128, 0, 128), -1)
        cv2.putText(frame, "FPS: %d"%(fps), (10, 10+t_size[1]+10-3), cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2.0, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.imshow('img', frame)
        cv2.waitKey(5)