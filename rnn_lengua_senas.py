import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import HeNormal, GlorotNormal, LecunNormal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime
import random
import pickle
import hashlib
import re

# Ruta al dataset
DATASET_DIR = r"dataset"

# Dimensiones reducidas para ahorrar memoria
IMG_WIDTH = 64
IMG_HEIGHT = 64
CHANNELS = 3

# Archivo de caché para guardar el dataset procesado
CACHE_FILE = "dataset_cache.pkl"

# 1. Generar hash del dataset para detectar cambios
def generar_hash_dataset():
    hash_md5 = hashlib.md5()
    
    # Obtener todas las rutas de archivos de imagen
    archivos = []
    clases = os.listdir(DATASET_DIR)
    
    for clase in sorted(clases):  # sorted para consistencia
        ruta_clase = os.path.join(DATASET_DIR, clase)
        if not os.path.isdir(ruta_clase):
            continue
        for archivo in sorted(os.listdir(ruta_clase)):
            if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                archivos.append(os.path.join(ruta_clase, archivo))
    
    # Hash basado en las rutas y timestamps de modificación
    for archivo in archivos:
        hash_md5.update(archivo.encode())
        hash_md5.update(str(os.path.getmtime(archivo)).encode())
    
    return hash_md5.hexdigest()

# 2. Guardar dataset en caché para evitar recargas innecesarias
def guardar_cache(X, y, dataset_hash):
    cache_data = {
        'X': X,
        'y': y,
        'hash': dataset_hash,
        'timestamp': datetime.now(),
        'img_dims': (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
    }
    
    print("Guardando dataset en caché...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Caché guardado en: {CACHE_FILE}")

# 3. Cargar dataset desde caché si existe y es válido
def cargar_cache():
    if not os.path.exists(CACHE_FILE):
        return None, None, None
    
    try:
        print("Verificando caché existente...")
        with open(CACHE_FILE, 'rb') as f:
            cache_data = f.read()
        
        # Verificar que el archivo no esté corrupto
        cache_data = pickle.loads(cache_data)
        
        # Verificar las dimensiones
        if cache_data['img_dims'] != (IMG_HEIGHT, IMG_WIDTH, CHANNELS):
            print("Dimensiones de imagen cambiaron, regenerando caché...")
            return None, None, None
        
        return cache_data['X'], cache_data['y'], cache_data['hash']
    
    except Exception as e:
        print(f"Error leyendo caché: {e}")
        print("Regenerando caché...")
        return None, None, None

# 4. Cargar imágenes y etiquetas
def cargar_dataset_completo():
    print("Cargando imágenes desde disco...")
    X = []
    y = []
    clases = os.listdir(DATASET_DIR)
    total_archivos = 0
    
    # Contar total de archivos para barra de progreso
    for clase in clases:
        ruta_clase = os.path.join(DATASET_DIR, clase)
        if not os.path.isdir(ruta_clase):
            continue
        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                total_archivos += 1
    
    print(f"Total de imágenes a procesar: {total_archivos}")
    
    procesados = 0
    for clase in clases:
        ruta_clase = os.path.join(DATASET_DIR, clase)
        if not os.path.isdir(ruta_clase):
            continue
        
        print(f"Procesando clase: {clase}")
        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(ruta_clase, archivo)
                try:
                    imagen = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    imagen_array = img_to_array(imagen)
                    X.append(imagen_array)
                    y.append(clase)
                    
                    procesados += 1
                    if procesados % 100 == 0:  # Mostrar progreso cada 100 imágenes
                        print(f"   Progreso: {procesados}/{total_archivos} ({procesados/total_archivos*100:.1f}%)")
                        
                except Exception as e:
                    print(f"Error cargando imagen {img_path}: {e}")

    X = np.array(X, dtype='float32') / 255.0  # Normalizar y convertir a float32
    y = np.array(y)
    
    print(f"Carga completada: {len(X)} imágenes procesadas")
    return X, y

# 5. Cargar dataset con caché inteligente
def cargar_dataset():
    print("Verificando caché del dataset...")
    
    # Generar hash actual del dataset
    hash_actual = generar_hash_dataset()
    
    # Intentar cargar desde caché
    X_cache, y_cache, hash_cache = cargar_cache()
    
    if X_cache is not None and hash_cache == hash_actual:
        print(f"Datos cargados: {len(X_cache)} imágenes")
        return X_cache, y_cache
    else:
        if X_cache is not None:
            print("Dataset modificado, recargando...")
        else:
            print("Caché no encontrado, cargando por primera vez...")
        
        # Cargar dataset completo
        X, y = cargar_dataset_completo()
        
        # Guardar en caché para futuras ejecuciones
        guardar_cache(X, y, hash_actual)
        
        return X, y

# Función para limpiar caché
def limpiar_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        print(f"Caché eliminado: {CACHE_FILE}")
    else:
        print("ℹNo hay caché para eliminar")

# Solo encuentra el siguiente número de modelo disponible
def obtener_siguiente_numero_modelo():
    # Crear carpeta principal si no existe
    if not os.path.exists("modelos"):
        os.makedirs("modelos")
        print("Carpeta 'modelos' creada")
        return 1
    
    # Buscar todos los números de modelo existentes
    numeros_existentes = []
    
    try:
        contenido = os.listdir("modelos")
        patron = re.compile(r'^modelo_(\d+)$')
        
        for item in contenido:
            ruta_completa = os.path.join("modelos", item)
            if os.path.isdir(ruta_completa):
                match = patron.match(item)
                if match:
                    numero = int(match.group(1))
                    numeros_existentes.append(numero)
        
        if numeros_existentes:
            siguiente_numero = max(numeros_existentes) + 1
            print(f"Modelos existentes: {sorted(numeros_existentes)}")
            print(f"Siguiente número de modelo: {siguiente_numero}")
            return siguiente_numero
        else:
            print("No se encontraron modelos existentes")
            print("Iniciando con modelo número: 1")
            return 1
            
    except Exception as e:
        print(f"Error al buscar modelos existentes: {e}")
        print("Iniciando con modelo número: 1")
        return 1

# Crear carpeta para el modelo con validación
def crear_carpeta_modelo(numero_modelo):
    carpeta_modelo = os.path.join("modelos", f"modelo_{numero_modelo}")
    
    if os.path.exists(carpeta_modelo):
        print(f"ADVERTENCIA: La carpeta {carpeta_modelo} ya existe")
        print(f"    Los archivos existentes serán sobrescritos")
        try:
            archivos_existentes = os.listdir(carpeta_modelo)
            if archivos_existentes:
                print(f"    Archivos existentes: {archivos_existentes}")
        except:
            pass
    else:
        os.makedirs(carpeta_modelo)
        print(f"Carpeta creada: {carpeta_modelo}")
    
    return carpeta_modelo

# Verificar integridad del sistema de modelos y muestra estadísticas
def verificar_sistema_modelos():
    print("\nVERIFICANDO SISTEMA DE MODELOS...")
    print("="*50)
    
    if not os.path.exists("modelos"):
        print("Carpeta 'modelos' no existe")
        return
    
    try:
        contenido = os.listdir("modelos")
        carpetas_modelo = []
        archivos_sueltos = []
        
        patron = re.compile(r'^modelo_(\d+)$')
        
        for item in contenido:
            ruta_completa = os.path.join("modelos", item)
            if os.path.isdir(ruta_completa):
                match = patron.match(item)
                if match:
                    numero = int(match.group(1))
                    carpetas_modelo.append((numero, item))
                else:
                    archivos_sueltos.append(item)
            else:
                archivos_sueltos.append(item)
        
        # Ordenar por número
        carpetas_modelo.sort(key=lambda x: x[0])
        
        print(f"ESTADÍSTICAS:")
        print(f"   • Total de modelos encontrados: {len(carpetas_modelo)}")
        
        if carpetas_modelo:
            numeros = [x[0] for x in carpetas_modelo]
            print(f"   • Rango de números: {min(numeros)} - {max(numeros)}")
            print(f"   • Números existentes: {numeros}")
            
            # Verificar huecos en la numeración
            numeros_faltantes = []
            for i in range(min(numeros), max(numeros) + 1):
                if i not in numeros:
                    numeros_faltantes.append(i)
            
            if numeros_faltantes:
                print(f"   • Números faltantes: {numeros_faltantes}")
            else:
                print(f"   • Numeración continua")
        
        if archivos_sueltos:
            print(f"   • Archivos/carpetas no reconocidos: {archivos_sueltos}")
        
        print(f"   • Próximo número: {obtener_siguiente_numero_modelo()}")
        
    except Exception as e:
        print(f"Error verificando sistema: {e}")

# 3. Preprocesamiento de datos
def preprocesar_datos(X, y):
    # Convertimos cada imagen (64x64x3) a secuencia (64, 64*3)
    X_seq = X.reshape((-1, IMG_HEIGHT, IMG_WIDTH * CHANNELS))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    return train_test_split(X_seq, y_cat, test_size=0.3, random_state=42), label_encoder

# 4. Crear modelo RNN
def crear_modelo(input_shape, num_classes, config):
    model = Sequential()
    
    # Agregar capas ocultas según la configuración
    for i in range(config['capas_ocultas']):
        if i == 0:
            # Primera capa con input_shape
            model.add(SimpleRNN(
                config['neuronas'], 
                activation=config['activacion'], 
                input_shape=input_shape,
                kernel_initializer=config['inicializador'],
                return_sequences=(i < config['capas_ocultas'] - 1)
            ))
        else:
            # Capas subsecuentes
            model.add(SimpleRNN(
                config['neuronas'], 
                activation=config['activacion'],
                kernel_initializer=config['inicializador'],
                return_sequences=(i < config['capas_ocultas'] - 1)
            ))
        model.add(Dropout(0.3))
    
    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=config['optimizador'],
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 5. Calcular métricas adicionales
def calcular_metricas(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, average='weighted'
    )
    
    return precision, recall, f1

# 6. Guardar información del modelo
def guardar_info_modelo(config, metricas, duracion, numero_modelo, carpeta_modelo):
    nombre_archivo = os.path.join(carpeta_modelo, f"modelo_{numero_modelo}.txt")
    
    with open(nombre_archivo, 'w', encoding='utf-8') as file:
        file.write(f"=== MODELO {numero_modelo} ===\n")
        file.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        file.write("ARQUITECTURA DE RED:\n")
        file.write(f"• Número de capas ocultas: {config['capas_ocultas']}\n")
        file.write(f"• Número de neuronas por capa: {config['neuronas']}\n")
        file.write(f"• Función de activación: {config['activacion']}\n")
        file.write(f"• Inicialización de pesos: {config['inicializador_nombre']}\n\n")
        
        file.write("PARÁMETROS DE ENTRENAMIENTO:\n")
        file.write(f"• Épocas: {config['epocas']}\n")
        file.write(f"• Tamaño de lote: {config['batch_size']}\n")
        file.write(f"• Tasa de aprendizaje: {config['tasa_aprendizaje']}\n")
        file.write(f"• Optimizador: {config['optimizador_nombre']}\n")
        file.write(f"• Criterio de pérdida: Categorical Crossentropy\n\n")
        
        file.write("MÉTRICAS:\n")
        file.write(f"• Precisión (Accuracy): {metricas['accuracy']:.4f} ({metricas['accuracy']*100:.2f}%)\n")
        file.write(f"• Precisión (Precision): {metricas['precision']:.4f}\n")
        file.write(f"• Recall: {metricas['recall']:.4f}\n")
        file.write(f"• F1-Score: {metricas['f1']:.4f}\n")
        file.write(f"• Pérdida: {metricas['loss']:.4f}\n")
        file.write(f"• Tiempo de entrenamiento: {duracion:.2f} segundos\n\n")
        
        if metricas['accuracy'] >= 0.98:
            file.write("* MODELO DE ALTA PRECISIÓN (≥98%) *\n")
    
    print(f"Información guardada en: {nombre_archivo}")

# 7. Configuración del modelo
def obtener_configuracion_manual():
    return {
        'capas_ocultas': 3,                   
        'neuronas': 192,                     
        'activacion': 'gelu',                   
        'inicializador': HeNormal(),             
        'inicializador_nombre': 'HeNorm',
        'optimizador': Adam(learning_rate=0.0008, beta_1=0.9, beta_2=0.999),
        'optimizador_nombre': 'Adam-optimized',
        'tasa_aprendizaje': 0.0008,
        'epocas': 90,                        
        'batch_size': 32                     
    }

# 8. Entrenar modelo único
def entrenar_modelo(X_train, X_test, y_train, y_test, numero_modelo):
    # Crear carpeta del modelo
    carpeta_modelo = crear_carpeta_modelo(numero_modelo)
    print(f"Carpeta del modelo: {carpeta_modelo}")
    
    # Obtener configuración la configuración
    config = obtener_configuracion_manual()
    
    print(f"\n{'='*60}")
    print(f"ENTRENANDO MODELO {numero_modelo}")
    print(f"{'='*60}")
    print("CONFIGURACIÓN SELECCIONADA:")
    print(f"• Capas ocultas: {config['capas_ocultas']}")
    print(f"• Neuronas por capa: {config['neuronas']}")
    print(f"• Activación: {config['activacion']}")
    print(f"• Inicializador: {config['inicializador_nombre']}")
    print(f"• Optimizador: {config['optimizador_nombre']}")
    print(f"• Tasa de aprendizaje: {config['tasa_aprendizaje']}")
    print(f"• Épocas: {config['epocas']}")
    print(f"• Batch size: {config['batch_size']}")
    print(f"{'='*60}")

    try:
        # Crear y entrenar modelo
        model = crear_modelo((X_train.shape[1], X_train.shape[2]), y_train.shape[1], config)
        
        print(f"\nIniciando entrenamiento...")
        inicio = time.time()
        historia = model.fit(
            X_train, y_train,
            epochs=config['epocas'],
            batch_size=config['batch_size'],
            validation_data=(X_test, y_test),
            verbose=1
        )
        duracion = time.time() - inicio

        # Evaluar modelo
        print(f"\nEvaluando modelo...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        precision, recall, f1 = calcular_metricas(model, X_test, y_test)
        
        metricas = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': loss
        }
        
        # Mostrar resultados
        print(f"\n{'='*60}")
        print(f"RESULTADOS DEL MODELO {numero_modelo}")
        print(f"{'='*60}")
        print(f"• Precisión (Accuracy): {accuracy*100:.2f}%")
        print(f"• Precisión (Precision): {precision:.4f}")
        print(f"• Recall: {recall:.4f}")
        print(f"• F1-Score: {f1:.4f}")
        print(f"• Pérdida: {loss:.4f}")
        print(f"• Tiempo de entrenamiento: {duracion:.2f} segundos")

        # Guardar información del modelo
        guardar_info_modelo(config, metricas, duracion, numero_modelo, carpeta_modelo)

        # Guardar modelo
        nombre_modelo = os.path.join(carpeta_modelo, f"modelo_{numero_modelo}.h5")
        model.save(nombre_modelo)
        print(f"\nMODELO GUARDADO: {nombre_modelo}")
        
        # Mensaje para modelos de alta precisión
        if accuracy >= 0.98:
            print(f"Este modelo tiene precisión ≥ 98%")

        # Crear y guardar gráfica
        plt.figure(figsize=(15, 5))
        
        # Gráfica de precisión
        plt.subplot(1, 2, 1)
        plt.plot(historia.history['accuracy'], 'b-', label='Entrenamiento', linewidth=2)
        plt.plot(historia.history['val_accuracy'], 'r-', label='Validación', linewidth=2)
        plt.title(f'Precisión - Modelo {numero_modelo}', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Precisión')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfica de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(historia.history['loss'], 'b-', label='Entrenamiento', linewidth=2)
        plt.plot(historia.history['val_loss'], 'r-', label='Validación', linewidth=2)
        plt.title(f'Pérdida - Modelo {numero_modelo}', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        nombre_grafico = os.path.join(carpeta_modelo, f'modelo_{numero_modelo}.png')
        plt.savefig(nombre_grafico, dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado como: {nombre_grafico}")
        
        # Mostrar gráfico
        plt.show()
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"ERROR durante el entrenamiento: {str(e)}")
        return False

# 9. Función principal
def main():
    print("ENTRENADOR DE MODELOS RNN - CONFIGURACIÓN MANUAL")
    print("="*70)
    
    # Verificar argumentos de línea de comandos
    if len(os.sys.argv) > 1:
        if os.sys.argv[1] == '--limpiar-cache':
            limpiar_cache()
            return
        elif os.sys.argv[1] == '--verificar':
            verificar_sistema_modelos()
            return
        elif os.sys.argv[1] == '--help':
            print("\nCOMANDOS DISPONIBLES:")
            print("   python script.py                   # Entrenar nuevo modelo")
            print("   python script.py --limpiar-cache   # Limpiar caché del dataset")
            print("   python script.py --verificar       # Verificar sistema de modelos")
            print("   python script.py --help            # Mostrar esta ayuda")
            return
    
    # Verificar sistema de modelos
    verificar_sistema_modelos()
    
    # Obtener el siguiente número de modelo
    numero_modelo = obtener_siguiente_numero_modelo()
    print(f"\nSe entrenará el MODELO {numero_modelo}")
    
    print("\nCargando dataset...")
    inicio_carga = time.time()
    X, y = cargar_dataset()
    tiempo_carga = time.time() - inicio_carga
    
    (X_train, X_test, y_train, y_test), label_encoder = preprocesar_datos(X, y)
    
    print(f"Datos cargados en {tiempo_carga:.2f} segundos:")
    print(f"   • Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   • Prueba: {X_test.shape[0]} muestras")
    print(f"   • Clases: {len(label_encoder.classes_)}")
    print(f"   • Clases encontradas: {list(label_encoder.classes_)}")
    
    input(f"\nPresiona Enter para entrenar el MODELO {numero_modelo}...")
    
    # Entrenar el modelo
    exito = entrenar_modelo(X_train, X_test, y_train, y_test, numero_modelo)
    
    if exito:
        print(f"\nMODELO {numero_modelo} ENTRENADO EXITOSAMENTE")
        print(f"Archivos generados en modelos/modelo_{numero_modelo}/:")
        print(f"   • modelo_{numero_modelo}.txt (información detallada)")
        print(f"   • modelo_{numero_modelo}.png (gráficas)")
        print(f"   • modelo_{numero_modelo}.h5 (modelo guardado)")
        print(f"\nPara entrenar otro modelo, ejecuta el script nuevamente.")
        print(f"   El próximo será el MODELO {numero_modelo + 1}")
        print(f"   La carga será mucho más rápida gracias al caché!")
    else:
        print(f"\nError durante el entrenamiento del MODELO {numero_modelo}")

if __name__ == '__main__':
    main()