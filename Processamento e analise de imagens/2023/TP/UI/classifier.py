import tensorflow as tf

def classify_image(image_path):
    # Configuração para utilizar a GPU da Apple
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Carregamento do modelo MobileNetV3-Small pré-treinado
    base_model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Congelando os pesos do MobileNetV3-Small
    base_model.trainable = False

    # Criação da camada de segmentação
    segmentation_layer = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')

    # Combinando o MobileNetV3-Small com a camada de segmentação
    model = tf.keras.Sequential([
        base_model,
        segmentation_layer
    ])

    # Carregamento da imagem de teste e pré-processamento
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    preprocessed_image = tf.keras.applications.mobilenet_v3.preprocess_input(image)

    # Realização da classificação da imagem
    prediction = model.predict(preprocessed_image)
    class_index = tf.argmax(prediction, axis=-1)[0]

    # Carregamento do arquivo de classes
    classes_path = 'caminho/para/classes.txt'
    with open(classes_path, 'r') as f:
        classes = f.read().splitlines()

    # Obtenção da classe predita
    predicted_class = classes[class_index]

    return predicted_class

# Exemplo de uso da função para classificar uma imagem
#image_path = 'caminho/para/imagem.jpg'
#predicted_class = classify_image(image_path)
#print('Classe predita:', predicted_class)

