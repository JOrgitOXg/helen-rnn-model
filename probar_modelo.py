import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import os

# Ruta del modelo guardado
modelo_path = "modelo_rnn_0.9882.h5"  # Poner nombre del modelo guardado

# Ruta a la carpeta del dataset (se necesita para cargar clases)
DATASET_DIR = r"C:\Users\jorge\Desktop\helen-rnn-model\dataset"

# Tamaño de imagen (debe ser igual al usado en entrenamiento)
IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3

# 1. Cargar clases para decodificar salida
def cargar_clases():
    clases = []
    for clase in os.listdir(DATASET_DIR):
        if os.path.isdir(os.path.join(DATASET_DIR, clase)):
            clases.append(clase)
    clases.sort()  # Asegurar orden para label encoder
    return clases

# 2. Preprocesar imagen nueva igual que en entrenamiento
def procesar_imagen(img_path):
    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_seq = img_array.reshape((1, IMG_HEIGHT, IMG_WIDTH * CHANNELS))  # Forma que espera el modelo
    return img_seq

# 3. Cargar modelo y predecir
def predecir(modelo_path, img_path, clases):
    model = load_model(modelo_path)
    img_proc = procesar_imagen(img_path)
    prediccion = model.predict(img_proc)
    indice = np.argmax(prediccion)
    clase_predicha = clases[indice]
    probabilidad = prediccion[0][indice]
    print(f"Clase predicha: {clase_predicha} con probabilidad {probabilidad:.4f}")

if __name__ == "__main__":
    clases = cargar_clases()
    imagen_prueba = r"C:\Users\jorge\Desktop\helen-rnn-model\test_fail.jpg"  # Pon ruta a la imagen que quieres probar para predecir
    predecir(modelo_path, imagen_prueba, clases)
