from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import os
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # general window shit
        self.setGeometry(50,50, 700,600) #can alternatively do 10 10 900 600
        self.setWindowTitle('ASL to Text Translator')
        
        # Creating the text widget 
        centralWidget = QWidget()
        mainLayout = QVBoxLayout()

        # Widgets for the camera portion (insert code for the camera here)
        cameraPlaceHolder = QLabel("Camera goes here")
        topLayout = QHBoxLayout()
        topLayout.addStretch()
        topLayout.addWidget(CameraWidget())
        topLayout.addStretch()


        # The middle layout where the play button goes
        recordButton = QPushButton("Record")
        middleLayout = QHBoxLayout()
        middleLayout.addStretch()
        middleLayout.addWidget(recordButton)
        middleLayout.addStretch()

        # Widgets for the bottom layout
        label = QLabel("Translation")
        editor = QTextEdit()
        bottomLayout = QVBoxLayout()
        bottomLayout.addWidget(label)
        bottomLayout.addWidget(editor)


        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(middleLayout)
        mainLayout.addLayout(bottomLayout)
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.show()


class CameraWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super(CameraWidget, self).__init__(*args, **kwargs)
        
        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            pass #quit

        self.viewfinder = QCameraViewfinder()
        self.viewfinder.show()

        layout = QVBoxLayout()
        layout.addWidget(self.viewfinder)
        self.setLayout(layout)


        self.select_camera(0)
        self.setWindowTitle("ASL to Text Editor")
        # self.show()

    def select_camera(self, i):
        self.camera = QCamera(self.available_cameras[i])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
        self.camera.start()

        self.current_camera_name = self.available_cameras[i].description()
        self.save_seq = 0


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("ASL to Text Editor")
    window = MainWindow()
    app.exec_()
