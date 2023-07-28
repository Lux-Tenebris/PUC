import os
import io
import zipfile
import pytesseract
from PIL import Image
from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__, template_folder='templates')

# Configurações para o Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Altere o caminho para o executável do Tesseract OCR no seu sistema
tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'  # Altere o caminho para o diretório tessdata do Tesseract OCR no seu sistema

def get_params():
    params = ""
    params += "--psm 12"

    configParams = []
    def configParam(param, val):
      return "-c " + param + "=" + val

    configParams.append(("chop_enable", "T"))
    configParams.append(('use_new_state_cost','F'))
    configParams.append(('segment_segcost_rating','F'))
    configParams.append(('enable_new_segsearch','0'))
    configParams.append(('textord_force_make_prop_words','F'))
    configParams.append(('tessedit_char_blacklist', '}><L'))
    configParams.append(('textord_debug_tabfind','0'))
    params += " ".join([configParam(p[0], p[1]) for p in configParams])
    return params

def get_blurbs(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bitwise_not(cv2.adaptiveThreshold(img_gray, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 75, 10))

    kernel = np.ones((2,2),np.uint8)
    img_gray = cv2.erode(img_gray, kernel,iterations = 2)
    img_gray = cv2.bitwise_not(img_gray)
    
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    pruned_contours = []
    mask = np.zeros_like(img)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height, width, channel = img.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100 and area < ((height / 3) * (width / 3)):
            pruned_contours.append(cnt)

    # find contours for the mask for a second pass after pruning the large and small contours
    cv2.drawContours(mask, pruned_contours, -1, (255,255,255), 1)
    
    contours2, hierarchy2 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    final_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)

    blurbs = []
    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area > 1000 and area < ((height / 3) * (width / 3)):
            draw_mask = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_BGR2GRAY)
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            cv2.fillPoly(draw_mask, [approx], (255,0,0))
            cv2.fillPoly(final_mask, [approx], (255,0,0))
            image = cv2.bitwise_and(draw_mask, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            y = approx[:, 0, 1].min()
            h = approx[:, 0, 1].max() - y
            x = approx[:, 0, 0].min()
            w = approx[:, 0, 0].max() - x
            image = image[y:y+h, x:x+w]
            pil_image = Image.fromarray(image)

            text = pytesseract.image_to_string(pil_image, lang="por", config=get_params())
            if text:
                blurbs.append(text)

    return blurbs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Verifica se o arquivo foi enviado na requisição
    if 'file' not in request.files:
        return 'Nenhum arquivo enviado', 400

    file = request.files['file']
    
    # Verifica se o arquivo tem uma extensão válida
    if not file.filename.lower().endswith(('.zip', '.cbz')):
        return 'Arquivo inválido. Por favor, envie um arquivo ZIP ou CBZ', 400

    # Salva o arquivo ZIP no sistema de arquivos
    zip_filename = secure_filename(file.filename)
    file.save(zip_filename)

    # Extrai as imagens do arquivo ZIP
    zip_file = zipfile.ZipFile(zip_filename)

    image_list = []

    for file_name in zip_file.namelist():
        if file_name.endswith(('.jpg', '.png')):
            # extract the image from the zip file
            image_data = zip_file.read(file_name)
            # create a PIL image object from the image data
            image = Image.open(io.BytesIO(image_data))
            # add the image to the list
            image_list.append(image)

    # Executa o OCR treinado nas imagens
    extracted_text = []

    # Processamento da extração de texto de forma sequencial
    for image in image_list:
        blurbs = get_blurbs(np.array(image))
        extracted_text.extend(blurbs)
    
    # Cria um arquivo de texto com o texto extraído
    output_filename = 'output.txt'
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join(extracted_text))
    
    # Retorna o arquivo de texto para o cliente
    return render_template('results.html', extracted_text=extracted_text)

# Main
if __name__ == '__main__':
    app.run()
