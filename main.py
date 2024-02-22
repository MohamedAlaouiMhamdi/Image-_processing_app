import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from imagepreApp import Ui_ImageProcessingAPP

class MyApp(QMainWindow, Ui_ImageProcessingAPP):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())