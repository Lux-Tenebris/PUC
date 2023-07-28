import cv2
import numpy as np

def remove_tags(image):
    contornos, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    if len(contornos) > 1:
        # Obtém o segundo maior contorno
        segundo_maior_contorno = contornos[0]

        # Desenha o contorno na imagem original
        cv2.drawContours(image, [segundo_maior_contorno], 0, (0, 0, 0), cv2.FILLED)

    return image

image = cv2.imread("d_left_mlo (2).png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = remove_tags(image)
cv2.imwrite("saida.png", image)