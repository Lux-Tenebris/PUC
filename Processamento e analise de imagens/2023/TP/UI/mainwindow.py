from PyQt5 import QtWidgets, uic
from image import Image
from PyQt5.QtWidgets import QFileDialog
import sys
import os
import cv2
import numpy as np
from frequencyFilterOperation import FrequencyFilterOperation
from classifier import classify_image

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('mainwindow.ui', self) # Load the .ui file
        self.originalImage  = ""
        self.label_resultado.setText(" ")
        self.label_imagem.setText("Nenhuma imagem selecionada")
        self.botao_buscarImagem.clicked.connect(self.openFileNameDialog)
        self.slide_escala.valueChanged.connect(self.updateEscala)
        self.slide_angulo.valueChanged.connect(self.updateAngulo)
        self.botao_aplicar.clicked.connect(self.aplicar)
        self.botao_aplicar2.clicked.connect(self.aplicar)
        self.botao_classificar.clicked.connect(self.classificar)
        self.show() # Show the GUI

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Imagem PNG (*.PNG);;Imagem JPEG (*.JPG);;All Files (*)", options=options)
        if fileName:
            self.writeImgSrc(fileName)
            self.originalImage = fileName

    def writeImgSrc(self, imagePath):
        if os.path.exists(imagePath):
            self.label_imagem.setPixmap(Image.readImgSrc(imagePath))

    def updateEscala(self):
        self.valor_escala.setText(str(float(self.slide_escala.value()) / 10.0))

    def updateAngulo(self):
        self.valor_angulo.setText(str(self.slide_angulo.value()))

    def aplicar(self):
        if self.originalImage == "":
            return
        image = cv2.imread(self.originalImage)
        height = image.shape[0]
        width = image.shape[1]
        
        espacial = self.combo_filtrosEspaciais.currentIndex()

        if espacial == 0:
            pass
        elif espacial == 1: # blur normal
            image = cv2.blur(image, (21, 21), 50)
        elif espacial == 0: # blur normal
            image = cv2.GaussianBlur(image, (21, 21), 50)
        elif espacial == 2: # blur normal
            image = cv2.medianBlur(image, 21)
        elif espacial == 3: # sobel
            image = cv2.Sobel(image, cv2.CV_8U,  0, 1, ksize=3)
        elif espacial == 4: # sobel
            image = cv2.Sobel(image, cv2.CV_8U,  1, 0, ksize=3)
        elif espacial == 5: # sobel
            image = cv2.Laplacian(image, cv2.CV_8U, 3)
        elif espacial == 5: # sobel
            image = cv2.Canny(image, 100, 200)

        frequencia = self.combo_filtrosFrequencia.currentIndex()
        options = self.combo_opcoes.currentIndex()

        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        middle = np.fft.fftshift(np.fft.fft2(image))

        if frequencia != 0:

            if frequencia == 1:
                if options == 0:
                    middlePassFilter = middle * FrequencyFilterOperation.idealHightPass(50, image.shape)
                elif options == 1:
                    middlePassFilter = middle * FrequencyFilterOperation.butterworthHightPass(50, image.shape, 10)
                elif options == 2:
                    middlePassFilter = middle * FrequencyFilterOperation.gaussianoHightPass(50, image.shape)
            elif frequencia == 2:
                if options == 0:
                    middlePassFilter = middle * FrequencyFilterOperation.idealLowPass(50, image.shape)
                elif options == 1:
                    middlePassFilter = middle * FrequencyFilterOperation.butterworthLowPass(50, image.shape, 10)
                elif options == 2:
                    middlePassFilter = middle * FrequencyFilterOperation.gaussianoLowPass(50, image.shape)
            passFilter = np.fft.ifftshift(middlePassFilter)
            reversePassFilter = np.fft.ifft2(passFilter)
            image = np.array(np.abs(reversePassFilter), dtype=np.uint8)

        if self.check_escala.isChecked():
            image = cv2.resize(image, 
                               None, 
                               fx=float(self.slide_escala.value() / 10.0), 
                               fy=float(self.slide_escala.value() / 10.0), 
                               interpolation=cv2.INTER_CUBIC)
        if self.check_angulo.isChecked():
            matrix = cv2.getRotationMatrix2D((width/2, height/2), 
                                             self.slide_angulo.value(), 
                                             1)
            image = cv2.warpAffine(image, matrix, (width, height))
        if self.check_espelhamento.isChecked():
            if self.radio_espelhamentoH.isChecked():
                image = cv2.flip(image, 1)
            elif self.radio_espelhamentoV.isChecked():
                image = cv2.flip(image, 0)
            elif self.radio_espelhamentoHV.isChecked():
                image = cv2.flip(image, -1)
        if self.check_translacao.isChecked():
            if self.radio_translacaoCE.isChecked():
                displacement = np.float32([[1, 0, -50], [0, 1, -90]])
                image = cv2.warpAffine(image, displacement, (width, height))
            elif self.radio_translacaoBD.isChecked():
                displacement = np.float32([[1, 0, 25], [0, 1, 50]])
                image = cv2.warpAffine(image, displacement, (width, height))
            
        cv2.imwrite("temp.png", image)
        self.writeImgSrc("temp.png")
        self.ready = True

    def classificar(self):
        if self.ready:
            classe = classify_image("temp.png")
            self.label_resultado.setText(str(classe))

app = QtWidgets.QApplication(sys.argv)
window = Ui()
app.exec_()
    