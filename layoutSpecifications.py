import sys
from PyQt5.QtWidgets import QHBoxLayout,QAction, QVBoxLayout, QLabel, QPushButton,QMainWindow, QApplication, QTextEdit, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # general window shit
        self.setGeometry(50,50, 500,300)
        self.setWindowTitle('ASL to Text Translator')
        
        # Creating the text widget 
        centralWidget = QWidget()
        mainLayout = QVBoxLayout()

        # Widgets for the camera portion (insert code for the camera here)
        cameraPlaceHolder = QLabel("Camera goes here")
        topLayout = QHBoxLayout()
        topLayout.addStretch()
        topLayout.addWidget(cameraPlaceHolder)
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


# Main function for running the app
def run():
    my_app = QApplication([])
    GUI = MainWindow()
    sys.exit(my_app.exec_())
run()

