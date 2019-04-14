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
import time
from PIL import ImageOps, Image

# Importing system specific utilities
import os
import sys

import breeze_resources

from threading import Timer

from imutils.video import VideoStream
from imutils.video import FPS
from collections import deque
import argparse
import imutils

import datetime

undoStack = []

FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 18, 24, 36, 48, 64, 72, 96, 144, 288]



currentTime = None



class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.selectedButton = 0
        
        # Dark Mode
        file = QFile(":/dark.qss")
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        self.setStyleSheet(stream.readAll())

        # Window geometry
        self.setGeometry(50,50, 600,400) #can alternatively do 10 10 900 600
        self.setWindowTitle('ASL to Text Translator')
        
        # Creating the text widget 
        centralWidget = QWidget()
        mainLayout = QVBoxLayout()

        # Widgets for the camera portion (insert code for the camera here)
        self.camera = VideoWidget()
        topLayout = QHBoxLayout()
        topLayout.addStretch()
        topLayout.addWidget(self.camera)
        topLayout.addStretch()

        # The middle layout where the play button goes
        self.recordButton = QPushButton()
        self.recordButton.setIcon(QIcon(QPixmap("./images/record-icon.png")))
        self.recordButton.setIconSize(QSize(60, 60))
        self.recordButton.setStyleSheet('QPushButton{border: 0px solid;}')
        self.recordButton.clicked.connect(self.changeButton)
        self.stopButton = QPushButton()
        self.stopButton.setIcon(QIcon(QPixmap("./images/stop-icon.png")))
        self.stopButton.setIconSize(QSize(60, 60))
        self.stopButton.setStyleSheet('QPushButton{border: none, outline: none;}')
        self.stopButton.clicked.connect(self.changeButton)
        middleLayout = QHBoxLayout()
        middleLayout.addStretch()
        middleLayout.addWidget(self.recordButton)
        middleLayout.addWidget(self.stopButton)
        self.stopButton.hide()
        middleLayout.addStretch()

        

        # Widgets for the bottom layout
        label = QLabel("Translation")
        self.editor = QTextEdit()
        self.editor.setFontPointSize(30)
        bottomLayout = QVBoxLayout()
        bottomLayout.addWidget(label)
        bottomLayout.addWidget(self.editor)

        # Generates menu bar
        edit_toolbar = QToolBar("Edit")
        edit_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(edit_toolbar)

        # Creating the undo menu bar
        undo_action = QAction(QIcon(os.path.join('images', 'arrow-curve-180-left.png')), "Undo", self)
        undo_action.setStatusTip("Undo last change")
        undo_action.triggered.connect(self.undoChar)
        edit_toolbar.addAction(undo_action)

        # Creating the redo on the menu bar
        redo_action = QAction(QIcon(os.path.join('images', 'arrow-curve.png')), "Redo", self)
        redo_action.setStatusTip("Redo last change")
        redo_action.triggered.connect(self.redoChar)
        edit_toolbar.addAction(redo_action)

        
        # Widget to allow us to edit text size
        format_toolbar = QToolBar("Format")
        format_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(format_toolbar)

        self.fonts = QFontComboBox()
        self.fonts.currentFontChanged.connect(self.updateTextBox)
        format_toolbar.addWidget(self.fonts)

        self.fontsize = QComboBox()
        self.fontsize.addItems([str(s) for s in FONT_SIZES])

        self.fontsize.currentIndexChanged[str].connect(self.updateTextBox)
        format_toolbar.addWidget(self.fontsize)   

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(middleLayout)
        mainLayout.addLayout(bottomLayout)
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)
        self.show()

    @pyqtSlot(str)
    def onUpdateText(self, text):
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.editor.setTextCursor(cursor)
        self.editor.ensureCursorVisible()

    def undoChar(self):
        if (len(self.editor.toPlainText()) > 0):
            lastChar = self.editor.toPlainText()[-1]
            undoStack.append(lastChar)
            self.editor.textCursor().deletePreviousChar()

    def redoChar(self):
        if len(undoStack) > 0:
            self.editor.textCursor().insertText(undoStack[0])
            undoStack.pop(0)
            
    @pyqtSlot()
    def changeButton(self):
        try:
            if (self.selectedButton == 0):
                self.selectedButton = 1
                self.recordButton.hide()
                self.stopButton.show()
            else:
                self.selectedButton = 0
                self.stopButton.hide()
                self.recordButton.show()
        except Exception as e:
            print (e)
    @pyqtSlot()
    def updateTextBox(self):
        self.editor.setCurrentFont(self.fonts.currentFont())
        self.editor.setFontPointSize(float(self.fontsize.currentText()))
        currentText = self.editor.toPlainText()
        self.editor.clear()
        self.editor.textCursor().insertText(currentText)
    
    # I'm ashamed i wrote this
    def hackFunction(self):
        print("Hello World")
        # b4First = Timer(3.0, self.onUpdateText, (''))
        # b4First.start()

        # first = Timer(6.0, self.onUpdateText, ('H'))
        # first.start()

        # second = Timer(9.0, self.onUpdateText, ('e'))
        # second.start()

        # third = Timer(12.0, self.onUpdateText, ('l'))
        # third.start()
        
        # fourth = Timer(15.0, self.onUpdateText, ('l'))
        # fourth.start()
                
        # fifth = Timer(18.0, self.onUpdateText, ('o'))
        # fifth.start()

        # sixth = Timer(21.0, self.onUpdateText, (' '))
        # sixth.start()

        # seventh = Timer(24.0, self.onUpdateText, ('W'))
        # seventh.start()

        # eigth = Timer(27.0, self.onUpdateText, ('o'))
        # eigth.start()

        # ninth = Timer(30.0, self.onUpdateText, ('r'))
        # ninth.start()

        # tenth = Timer(33.0, self.onUpdateText, ('l'))
        # tenth.start()
        
        # eleventh = Timer(36.0, self.onUpdateText, ('d'))
        # eleventh.start()


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


class CV2Video(QObject):
    # Defining signals to be sent out 

    # VideoSignal sends out an image to be fed into VideoWidget
    # TextSignal sends out a string to be sent out 
    VideoSignal = pyqtSignal(QImage)
    TextSignal = pyqtSignal(str)

    def __init__(self):
        super(CV2Video, self).__init__()

    @pyqtSlot()
    def startVideo(self):
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
            start_time = time.time()
            letters = ["A", "B", "C", "D", "G", "I", "L", "V", "Y"]
            cap = cv2.VideoCapture(0)
            while 1:
                ret, img = cap.read()
                if ret:
                    # img, top left corner, bottom right corner
                    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
                    crop_img = img[100:300, 100:300]
                    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    value = (35, 35)
                    blurred = cv2.GaussianBlur(grey, value, 0)
                    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)
                    img = cv2.flip(img, 1) # To flip image

                    color_swapped_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    height, width, _ = color_swapped_image.shape
                    qt_image = QImage(color_swapped_image.data,width, height, color_swapped_image.strides[0],QImage.Format_RGB888)
                    self.VideoSignal.emit(qt_image)

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
                    elapsed_time = time.time() - start_time
                    # print(testY)
                    # Print predicted letter, only if it has changed since the last prediction
                    for i in range(len(testY[0])):
                        if (testY[0][i] != testY_previous[0][i]) and (elapsed_time > 2):
                            if testY[0][i] == [1]:
                                start_time = time.time()
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    continue
            
            cap.release()
            cv2.destroyAllWindows()


class VideoWidget(QWidget):
    def __init__(self):
        super(VideoWidget, self).__init__()
        # initially has no image
        self.image = QImage()
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0,0, self.image)
        self.image = QImage()

    def initUI(self):
        self.setWindowTitle('Built Myself')
    
    @pyqtSlot(QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer dropped frame")
        #set the image to be the image received as an argument in this function 
        self.image= image
        if image.size != self.size():
            self.setFixedSize(image.size())
        self.update() #update after we've resized the image





# ===========================
# ===========================


class UtilVideo(QObject):
    VideoSignal = pyqtSignal(QImage)
    startHack = pyqtSignal()
    def __init__(self):
        super(UtilVideo, self).__init__()

    @pyqtSlot()
    def startVideo(self):
        
        currentTime  = int(datetime.datetime.now().time().strftime('%S'))
        
        print(currentTime)
        # Construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-t", "--tracker", type=str, default="csrt",
            help="OpenCV object tracker type")
        ap.add_argument("-b", "--buffer", type=int, default=12,
            help="Max buffer size")
        ap.add_argument("-s", "--serial", type=int, default=0,
            help="Set to false if not connected to Arduino")
        args = vars(ap.parse_args())

        # Extract the OpenCV version info
        (major, minor) = cv2.__version__.split(".")[:2]

        # If we are using OpenCV 3.2 OR BEFORE, we can use a special factory
        # function to create our object tracker
        if int(major) == 3 and int(minor) < 3:
            tracker = cv2.Tracker_create(args["tracker"].upper())

        # Otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
        # approrpiate object tracker constructor:
        else:
            # Initialize a dictionary that maps strings to their corresponding
            # OpenCV object tracker implementations
            OPENCV_OBJECT_TRACKERS = {
                "csrt": cv2.TrackerCSRT_create, # more accurate but slower than KCF
                "kcf": cv2.TrackerKCF_create, # doesn't handle full occlusion well
                "boosting": cv2.TrackerBoosting_create, # bad
                "mil": cv2.TrackerMIL_create, # bad
                "tld": cv2.TrackerTLD_create, # false-positives
                "medianflow": cv2.TrackerMedianFlow_create, # not good with motion jumps
                "mosse": cv2.TrackerMOSSE_create # VERY fast, not as accurate
            }

            # Grab the appropriate object tracker using our dictionary of
            # OpenCV object tracker objects
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

        # Initialize the bounding box coordinates of the object we are going to track
        initBB = None

        # Grab the reference to the web cam
        print("[INFO] starting video stream...")
        #vs = VideoStream('/dev/video1').start()
        vs = VideoStream(src=0).start()
        time.sleep(1.0)

        # Initialize the list of tracked points, the frame counter,
        # and the coordinate deltas

        # Initialize the FPS throughput estimator
        fps = None

    
        # Loop over frames from the video stream
        while True:
            # Grab the current frame
            frame = vs.read()
            frame = cv2.flip(frame, 1)
            # Check to see if we have reached the end of the stream
            if frame is None:
                break
            # Resize the frame (so we can process it faster) and grab the frame dimensions
            frame = imutils.resize(frame, width=650)
            (H, W) = frame.shape[:2]
            


            timer = str ((currentTime - int(datetime.datetime.now().time().strftime('%S'))  ) % 3)


            
            #center_frame = (H / 2 , W / 2)
            center = None

            # Check to see if we are currently tracking an object
            if initBB is not None:

                # Grab the new bounding box coordinates of the object
                (success, box) = tracker.update(frame)
                # Check to see if the tracking was a success
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2) 
                    cv2.putText(frame, timer, 
                    (x,y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (255,255,255),
                    2)
                                        # Fix this to be the center of the box
                    center = (int( x + w / 2 ), int( y + h / 2))
                    #pts.appendleft(center)
                 # To flip image

                color_swapped_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = color_swapped_image.shape
                qt_image = QImage(color_swapped_image.data,width, height, color_swapped_image.strides[0],QImage.Format_RGB888)
                self.VideoSignal.emit(qt_image)


                # Update the FPS counter
                fps.update()
                fps.stop()

            # Show the output frame

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        


            # If the 's' key is selected, we are going to "select" a bounding
            # box to track
            if key == ord("s"):
                # SSlect the bounding box of the object we want to track (make
                # sure you press ENTER or SPACE after selecting the ROI)
                initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                    showCrosshair=True)

                # Start OpenCV object tracker using the supplied bounding box
                # coordinates, then start the FPS throughput estimator as well
                self.startHack.emit()
                tracker.init(frame, initBB)
                fps = FPS().start()
                cv2.destroyAllWindows()
                


            # If the `q` key was pressed, break stop the robot and from the loop
            elif key == ord("q"):
                if (args["serial"] == 0):
                    ser.write('0')
                    print("STOP")
                time.sleep(3)
                break

        # Release webcam pointer
        cv2.destroyAllWindows()
        vs.stop()
        # Close all windows
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("ASL to Text Editor")

    # create the open cv camera and move it to another thread
    newThread = QThread()
    newThread.start()
    cv2Feed = UtilVideo()
    cv2Feed.moveToThread(newThread)

    window = MainWindow()

    cv2Feed.VideoSignal.connect(window.camera.setImage)
    # cv2Feed.TextSignal.connect(window.onUpdateText)
    window.recordButton.clicked.connect(cv2Feed.startVideo)
    cv2Feed.startHack.connect(window.hackFunction)


 
    app.exec_()


