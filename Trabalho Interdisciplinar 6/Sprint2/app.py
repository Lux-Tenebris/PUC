import os
import zipfile
import pytesseract
from PIL import Image
from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configurações para o Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Altere o caminho para o executável do Tesseract OCR no seu sistema
tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'  # Altere o caminho para o diretório tessdata do Tesseract OCR no seu sistema

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
    if not file.filename.lower().endswith('.zip'):
        return 'Arquivo inválido. Por favor, envie um arquivo ZIP', 400

    # Salva o arquivo ZIP no sistema de arquivos
    zip_filename = secure_filename(file.filename)
    file.save(zip_filename)

    # Extrai as imagens do arquivo ZIP
    extracted_images = []
    with zipfile.ZipFile(zip_filename) as zip_file:
        for entry in zip_file.infolist():
            with zip_file.open(entry) as image_file:
                image = Image.open(image_file)
                extracted_images.append(image)

    # Executa o Tesseract OCR nas imagens
    extracted_text = []
    for image in extracted_images:
        text = pytesseract.image_to_string(image, config=tessdata_dir_config)
        extracted_text.append(text)

    # Cria um arquivo de texto com o texto extraído
    output_filename = 'output.txt'
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join(extracted_text))

    # Retorna o arquivo de texto para o cliente
