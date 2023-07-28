import os
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore

class Image:

    @staticmethod
    def readImgSrc(filename):
        if os.path.exists(filename):
            return QPixmap(filename).scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        else:
               return None