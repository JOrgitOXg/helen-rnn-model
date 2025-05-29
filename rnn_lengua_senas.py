import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ruta al dataset
DATASET_DIR = r"C:\Users\jorge\Desktop\helen-rnn-model\dataset"

# Dimensiones reducidas para ahorrar memoria
IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3

# 1. Cargar imágenes y etiquetas con tamaño reducido
def cargar_dataset():
    X = []
    y = []
    clases = os.listdir(DATASET_DIR)

    for clase in clases:
        ruta_clase = os.path.join(DATASET_DIR, clase)
        if not os.path.isdir(ruta_clase):
            continue
        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(ruta_clase, archivo)
                try:
                    imagen = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    imagen_array = img_to_array(imagen)
                    X.append(imagen_array)
                    y.append(clase)
                except Exception as e:
                    print(f"Error cargando imagen {img_path}: {e}")

    X = np.array(X, dtype='float32') / 255.0  # Normalizar y convertir a float32
    y = np.array(y)

    return X, y

# 2. Preprocesamiento de datos
def preprocesar_datos(X, y):
    # Convertimos cada imagen (64x64x3) a secuencia (64, 64*3)
    X_seq = X.reshape((-1, IMG_HEIGHT, IMG_WIDTH * CHANNELS))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    return train_test_split(X_seq, y_cat, test_size=0.3, random_state=42), label_encoder

# 3. Crear modelo RNN
def crear_modelo(input_shape, num_classes, config):
    model = Sequential()
    model.add(SimpleRNN(config['neuronas'], activation=config['activacion'], input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=config['optimizador'],
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 4. Entrenamiento con múltiples hiperparámetros
def buscar_modelos(X_train, X_test, y_train, y_test):
    combinaciones = [
        {'neuronas': 64, 'activacion': 'tanh', 'optimizador': Adam(0.001), 'epocas': 50, 'batch_size': 32},
        {'neuronas': 128, 'activacion': 'relu', 'optimizador': RMSprop(0.0001), 'epocas': 100, 'batch_size': 64},
        {'neuronas': 256, 'activacion': 'tanh', 'optimizador': SGD(0.01), 'epocas': 150, 'batch_size': 32},
    ]

    for i, config in enumerate(combinaciones):
        print(f"\n--- Ejecutando modelo {i+1} ---")
        print("Parámetros:", config)

        model = crear_modelo((X_train.shape[1], X_train.shape[2]), y_train.shape[1], config)

        inicio = time.time()
        historia = model.fit(
            X_train, y_train,
            epochs=config['epocas'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            verbose=1
        )
        duracion = time.time() - inicio

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Precisión obtenida: {acc*100:.2f}%")
        print(f"Tiempo de ejecución: {duracion:.2f} segundos")

        if acc >= 0.98:
            nombre_modelo = f"modelo_rnn_{acc:.4f}.h5"
            model.save(nombre_modelo)
            print(f"Modelo guardado ({nombre_modelo}) - precisión > 98%")

        # Graficar precisión y pérdida
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(historia.history['accuracy'], label='Entrenamiento')
        plt.plot(historia.history['val_accuracy'], label='Validación')
        plt.title('Precisión')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(historia.history['loss'], label='Entrenamiento')
        plt.plot(historia.history['val_loss'], label='Validación')
        plt.title('Pérdida')
        plt.legend()
        plt.tight_layout()
        plt.show()

# 5. Ejecutar todo
def main():
    print("Cargando datos...")
    X, y = cargar_dataset()
    (X_train, X_test, y_train, y_test), label_encoder = preprocesar_datos(X, y)
    buscar_modelos(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
