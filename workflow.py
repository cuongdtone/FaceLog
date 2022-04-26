from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
import numpy as np
import traceback
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

from Face.face_threading import face_thread

from tab1 import tab1
from log import Log

class RecognThread(QThread):
    change_screen_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture("/dev/video1")
        thread = face_thread(cap)
        frame_final_queue, frame_ori_queue = thread.run()

        while self._run_flag:
            data = frame_final_queue.get()
            frame_ori = frame_ori_queue.get()

            # cv_img = data['frame']
            # info = data['people'][0]
            self.change_screen_signal.emit(data)
        cap.release()
        # shut down capture system

    def stop(self):
        self._run_flag = False
        self.wait()




class Tab1(QWidget, tab1):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Windown")

        self.display_width, self.display_height = 720, 480
        # create the video capture thread
        self.thread = RecognThread()
        # connect its signal to the update_image slot
        self.thread.change_screen_signal.connect(self.update_info)
        # start the thread
        self.thread.start()

        self.log = Log()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    @pyqtSlot(dict)
    def update_info(self, data):
        try:
            person = data['people'][0]

            sucess = self.log.login(person)

            if sucess:
                self.update_image(cv2.imread('Db/' + person['Name'] + '/1.jpg'), self.sub_screen, 200, 200)
                self.textBrowser.append(person['Name'])

            self.name.setText(person['Name'])
            self.position.setText(person['Position'])
        except Exception:
            print(traceback.format_exc())
            print(sys.exc_info()[2])
        # self.update_image(face, self.sub_screen)
        self.update_image(data['frame'], self.screen, self.display_width, self.display_height)
        # print(box)
    def update_image(self, cv_img, screen, display_width, display_height):
        qt_img = self.convert_cv_qt(cv_img, display_width, display_height)
        screen.setPixmap(qt_img)
    def convert_cv_qt(self, cv_img, disply_width, display_height):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        #rgb_image = cv2.flip(rgb_image, flipCode=1)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = Tab1()
    a.show()
    sys.exit(app.exec_())