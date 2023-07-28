#Dependencias
import os
import io
import zipfile
import pytesseract
from PIL import Image
from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
import multiprocessing

def processar_imagem(imagem) -> str:
    print(f"Thread {os.getpgid()} tabalhando agora")
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Altere o caminho para o executável do Tesseract OCR no seu sistema
    tessdata_dir_config = '--tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'  # Altere o caminho para o diretório tessdata do Tesseract OCR no seu sistema
    texto = pytesseract.image_to_string(imagem, config=tessdata_dir_config)
    return texto

app = Flask(__name__, template_folder='templates')

# Configurações para o Tesseract OCR


# Carrega o OCR treinado
#ocr_engine = pytesseract.pytesseract.Tesseract(trained_data_file='eng.traineddata')

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

    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            extracted = [pool.apply_async(processar_imagem, args=(x,)) for x in image_list]

            for texto in extracted:
                texto.wait()

            for texto in extracted:
                extracted_text.append(texto)
    # Processamento da extração de texto de forma sequencial
    """for image in image_list:
        #text = ocr_engine.image_to_string(image, lang='por')
        text = pytesseract.image_to_string(image, config=tessdata_dir_config)
        extracted_text.append(text)"""
    
    # Cria um arquivo de texto com o texto extraído
    output_filename = 'output.txt'
    with open(output_filename, 'w') as output_file:
        output_file.write('\n'.join(extracted_text))
    
    #Chama a correção de texto
    #correction = textCorrection.correctSpelling(extracted_text)

    # Retorna o arquivo de texto para o cliente
    return render_template('results.html', extracted_text=extracted_text)

#Main
if __name__ == '__main__':
    app.run()