from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np

from utils import save_new_image
import os
from Face.face import Face_Model
import time

face = Face_Model()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: break
            self.change_pixmap_signal.emit(frame)
        cap.release()
        # shut down capture system

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

from tab2 import tab2
class App(QWidget, tab2):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowTitle("Windown")
        self.disply_width = 640
        self.display_height = 480

        self.shot.clicked.connect(self.save_img)
        self.complete.clicked.connect(self.create_data)
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
    def create_data(self):
        name = self.name.text()
        position = self.position.text()

        face.create_data_file(name, position, '')

        QMessageBox.warning(self, 'Warning!', "Completed !")

    def save_img(self):
        if self.name.text() == '':
            QMessageBox.warning(self, 'Warning!', "Name field cannot emtpy!")
            return
        try:
            os.mkdir('Db/' + self.name.text())
        except:
            pass
        save_new_image('Db/' + self.name.text(), self.cv_img)


    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.cv_img = cv_img
        qt_img = self.convert_cv_qt(cv_img)
        self.screen.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        #rgb_image = cv2.flip(rgb_image, flipCode=1)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())