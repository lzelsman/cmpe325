# Importing all the gui elements
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import tensorflow as tf
import parameters as par
import cv2
import numpy as np
from PIL import ImageOps, Image

# Importing system specific utilities
import os
import sys

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
        recordButton = QPushButton()
        recordButton.setIcon(QIcon(QPixmap("./images/record-icon.png")))
        recordButton.setIconSize(QSize(60, 60))
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

    def onUpdateText(self, text):
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.editor.setTextCursor(cursor)
        self.editor.ensureCursorVisible()

    def recordHandler(self):
        #self.camera.startCamera()
        sys.stdout = Stream(newText=self.onUpdateText)
        saver = tf.train.import_meta_graph(par.saved_path + str('501.meta'))
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint('./Saved/'))

            # Get Operations to restore
            graph = sess.graph

            # Get Input Graph
            X = graph.get_tensor_by_name('Input:0')
            #Y = graph.get_tensor_by_name('Target:0')
            # keep_prob = tf.placeholder(tf.float32)
            keep_prob = graph.get_tensor_by_name('Placeholder:0')

            # Get Ops
            prediction = graph.get_tensor_by_name('prediction:0')
            logits = graph.get_tensor_by_name('logits:0')
            accuracy = graph.get_tensor_by_name('accuracy:0')

            # Get the image
            count = 0
            while 1:
                cap = cv2.VideoCapture(0)
                ret, img = cap.read()
                if ret:
                    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
                    crop_img = img[100:300, 100:300]
                    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    value = (35, 35)
                    blurred = cv2.GaussianBlur(grey, value, 0)
                    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    img = cv2.resize(img, dsize=None, fx=1.5, fy=1.5)
                    img = cv2.flip(img, 1) # To flip image
                    #cv2.imshow('title',thresh1)
                    cv2.imshow('frame',img)
                    thresh1 = (thresh1 * 1.0) / 255
                    thresh1 = Image.fromarray(thresh1)
                    thresh1 = ImageOps.fit(thresh1, [par.image_size, par.image_size])
                    if par.threshold:
                        testImage = np.reshape(thresh1, [-1, par.image_size, par.image_size, 1])
                    else:
                        testImage = np.reshape(thresh1, [-1, par.image_size, par.image_size, 3])
                    testImage = testImage.astype(np.float32)
                    if count == 0: # First iteration
                        testY = sess.run(prediction, feed_dict={X: testImage, keep_prob: 1.0})
                        testY_previous = testY
                    else: 
                        testY_previous = testY
                        testY = sess.run(prediction, feed_dict={X: testImage, keep_prob: 1.0})
                    count += 1
                    # print(testY)
                    # Print predicted letter, only if it has changed since the last prediction
                    for i in range(len(testY[0])):
                        if testY[0][i] != testY_previous[0][i]:
                            if testY[0][0] == [1]:
                                print("A")
                            elif testY[0][1] == [1]: 
                                print("B")
                            elif testY[0][2] == [1]:
                                print("C")
                            elif testY[0][3] == [1]:
                                print("D")
                            elif testY[0][4] == [1]:
                                print("G")
                            elif testY[0][5] == [1]:
                                print("I")
                            elif testY[0][6] == [1]:
                                print("L")
                            elif testY[0][7] == [1]:
                                print("V")
                            elif testY[0][8] == [1]:
                                print("Y")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    continue
            
            cap.release()
            cv2.destroyAllWindows()

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


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))  

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
