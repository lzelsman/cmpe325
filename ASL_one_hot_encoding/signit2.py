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

#app = QApplication(sys.argv)
#file = QFile(":/dark.qss")
#file.open(QFile.ReadOnly | QFile.Text)
#stream = QTextStream(file)
#app.setStyleSheet(stream.readAll())

undoStack = []

FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 18, 24, 36, 48, 64, 72, 96, 144, 288]

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
        self.recordButton.setStyleSheet('QPushButton{border: none, outline: none;}')
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
                    cv2.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
                    crop_img = img[100:300, 100:300]
                    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    value = (35, 35)
                    blurred = cv2.GaussianBlur(grey, value, 0)
                    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    img = cv2.resize(img, dsize=None, fx=1.5, fy=1.5)
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
                                # print(letters[i])

                                # emit if recognized
                                self.TextSignal.emit(str(letters[i]))
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("ASL to Text Editor")

    # create the open cv camera and move it to another thread
    newThread = QThread()
    newThread.start()
    cv2Feed = CV2Video()
    cv2Feed.moveToThread(newThread)

    window = MainWindow()

    cv2Feed.VideoSignal.connect(window.camera.setImage)
    cv2Feed.TextSignal.connect(window.onUpdateText)
    window.recordButton.clicked.connect(cv2Feed.startVideo)

 
    app.exec_()
