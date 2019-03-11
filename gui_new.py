# Importing all the gui elements
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

# Importing system specific utilities
import os
import sys

# OpenCV related imports
import cv2
import numpy

FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 18, 24, 36, 48, 64, 72, 96, 144, 288]

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(50,50, 700,600) #can alternatively do 10 10 900 600
        self.setWindowTitle('ASL to Text Translator')
        
        # Creating the text widget 
        centralWidget = QWidget()
        mainLayout = QVBoxLayout()

        # Widgets for the camera portion (insert code for the camera here)
        self.camera = OpenCVCamera()
        topLayout = QHBoxLayout()
        topLayout.addStretch()
        topLayout.addWidget(self.camera)
        topLayout.addStretch()


        # The middle layout where the play button goes
        recordButton = QPushButton("Record")
        middleLayout = QHBoxLayout()
        middleLayout.addStretch()
        middleLayout.addWidget(recordButton)
        middleLayout.addStretch()
        recordButton.clicked.connect(self.recordHandler)

        # Widgets for the bottom layout
        label = QLabel("Translation")
        self.editor = QTextEdit()
        bottomLayout = QVBoxLayout()
        bottomLayout.addWidget(label)
        bottomLayout.addWidget(self.editor)

        # Generates tmenu bar
        edit_toolbar = QToolBar("Edit")   
        edit_menu = self.menuBar().addMenu("&Edit")

        # Creating the undo menu bar
        undo_action = QAction(QIcon(os.path.join('images', 'arrow-curve-180-left.png')), "Undo", self)
        undo_action.setStatusTip("Undo last change")
        undo_action.triggered.connect(self.editor.undo)
        edit_menu.addAction(undo_action)

        # Creating the redo on the menu bar
        redo_action = QAction(QIcon(os.path.join('images', 'arrow-curve.png')), "Redo", self)
        redo_action.setStatusTip("Redo last change")
        redo_action.triggered.connect(self.editor.redo)
        edit_toolbar.addAction(redo_action)
        edit_menu.addAction(redo_action)

        
        # Widget to tallow us to edit text size
        format_toolbar = QToolBar("Format")
        format_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(format_toolbar)
        format_menu = self.menuBar().addMenu("&Format")

        # We need references to these actions/settings to update as selection changes, so attach to self.
        self.fonts = QFontComboBox()
        self.fonts.currentFontChanged.connect(self.editor.setCurrentFont)
        format_toolbar.addWidget(self.fonts)

        self.fontsize = QComboBox()
        self.fontsize.addItems([str(s) for s in FONT_SIZES])

        # Connect to the signal producing the text of the current selection. Convert the string to float
        # and set as the pointsize. We could also use the index + retrieve from FONT_SIZES.
        self.fontsize.currentIndexChanged[str].connect(lambda s: self.editor.setFontPointSize(float(s)) )
        format_toolbar.addWidget(self.fontsize)
        # Menu bar stuff    

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(middleLayout)
        mainLayout.addLayout(bottomLayout)
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.show()

    def recordHandler(self):
        self.camera.startCamera()

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


# Camera widget from openCV
class OpenCVCamera(QWidget):
    def __init__(self):
        super().__init__()
        self.camera = cv2.VideoCapture(0)
        self.cameraRunning = False
    
    def startCamera(self):
        self.running =True
        while(self.running):
            ret, feed = self.camera.read()
            grayscale = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)
            cv2.imshow('feed', grayscale)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("ASL to Text Editor")
    window = MainWindow()
    app.exec_()
