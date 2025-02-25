import cv2
import numpy as np
import os
import pickle
from scipy.spatial import distance
from skimage.feature import hog
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import time
import tkinter as tk

class CBIRIndexer:
    """
    Clase para la indexación de imágenes en un sistema CBIR.
    Se encarga de extraer y almacenar las características de las imágenes.
    """
    
    def __init__(self, feature_bins=(8, 8, 8), hog_cell_size=(16, 16), hog_block_size=(2, 2)):
        """
        Inicializa el indexador con parámetros para la extracción de características.
        
        Args:
            feature_bins: Número de bins para el histograma de color (H, S, V)
            hog_cell_size: Tamaño de celda para HOG
            hog_block_size: Tamaño de bloque para HOG
        """
        self.feature_bins = feature_bins
        self.hog_cell_size = hog_cell_size
        self.hog_block_size = hog_block_size
        self.image_features = {}
        self.hog_images = {}
        
    def extract_background(self, image):
        """
        Segmenta el fondo utilizando umbralización en el canal de valor (V) del espacio HSV.
        
        Args:
            image: Imagen BGR
            
        Returns:
            Imagen con el fondo segmentado
        """
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, thresholded = cv2.threshold(image_hsv[:, :, 2], 100, 255, cv2.THRESH_BINARY)
        return cv2.bitwise_and(image, image, mask=thresholded)
    
    
    def extract_color_histogram(self, image):
        """
        Extrae el histograma de color en el espacio HSV.
        
        Args:
            image: Imagen BGR
            
        Returns:
            Histograma de color normalizado
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, self.feature_bins, 
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def normalize_hog_features(self, hog_features):
        """
        Normaliza las características HOG para que estén en el rango [0, 1].
        
        Args:
            hog_features: Características HOG
            
        Returns:
            Características HOG normalizadas
        """
        return cv2.normalize(hog_features, None, 0, 1, cv2.NORM_MINMAX)
    
    def extract_hog_features(self, image):
        """
        Extrae características HOG de la imagen en escala de grises.
        Args:
            image: Imagen BGR
            
        Returns:
            Características HOG normalizadas y la imagen HOG
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features, hog_image = hog(gray, pixels_per_cell=self.hog_cell_size, 
                                 cells_per_block=self.hog_block_size, 
                                 visualize=True, feature_vector=True)
        features = self.normalize_hog_features(features)
        features = features.flatten()
        return features, hog_image
    
    def extract_features(self, image_path):
        """
        Carga una imagen y extrae sus características.
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Histograma de color, características HOG e imagen HOG
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: No se pudo cargar la imagen {image_path}")
            return None, None, None
            
        try:
            image = self.extract_background(image)
            color_hist = self.extract_color_histogram(image)
            hog_features, hog_image = self.extract_hog_features(image)
            return color_hist, hog_features, hog_image
        except Exception as e:
            print(f"Error durante la extracción de características: {e}")
            return None, None, None
    
    def index_folder(self, folder_path):
        """
        Indexa todas las imágenes en una carpeta.
        
        Args:
            folder_path: Ruta de la carpeta con imágenes
            
        Returns:
            True si la indexación fue exitosa, False en caso contrario
        """
        start_time = time.time()
        print(f"Iniciando indexación de imágenes en {folder_path}...")
        
        if not os.path.exists(folder_path):
            print(f"Error: La carpeta {folder_path} no existe")
            return False
            
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_count = 0
        failed_count = 0
        
        for file_name in os.listdir(folder_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in image_extensions:
                image_path = os.path.join(folder_path, file_name)
                try:
                    color_hist, hog_features, hog_image = self.extract_features(image_path)
                    
                    if color_hist is not None and hog_features is not None:
                        combined_features = np.hstack([color_hist, hog_features])
                        self.image_features[file_name] = combined_features
                        self.hog_images[file_name] = hog_image
                        image_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"Error procesando {file_name}: {e}")
                    failed_count += 1
                    
                if (image_count + failed_count) % 10 == 0:
                    print(f"Procesadas {image_count} imágenes (Fallidas: {failed_count})...")
        
        end_time = time.time()
        print(f"Indexación completada. Se procesaron {image_count} imágenes " 
              f"(Fallidas: {failed_count}) en {end_time - start_time:.2f} segundos")
        return image_count > 0
    
    def save_index(self, index_file):
        """
        Guarda el índice en un archivo.
        
        Args:
            index_file: Ruta del archivo donde se guardará el índice
            
        Returns:
            True si el guardado fue exitoso, False en caso contrario
        """
        try:
            index_data = {
                'features': self.image_features,
                'hog_images': self.hog_images,
                'params': {
                    'feature_bins': self.feature_bins,
                    'hog_cell_size': self.hog_cell_size,
                    'hog_block_size': self.hog_block_size
                }
            }
            
            with open(index_file, 'wb') as f:
                pickle.dump(index_data, f)
                
            print(f"Índice guardado correctamente en {index_file}")
            return True
        except Exception as e:
            print(f"Error al guardar el índice: {e}")
            return False






class CBIRSearcher:
    """
    Clase para la búsqueda de imágenes en un sistema CBIR.
    Utiliza el índice creado por CBIRIndexer para buscar imágenes similares.
    """
    def __init__(self, index_file=None):
        """
        Inicializa el buscador.
        
        Args:
            index_file: Ruta del archivo de índice (opcional)
        """
        self.indexer = None
        self.image_features = {}
        self.hog_images = {}
        
        if index_file and os.path.exists(index_file):
            self.load_index(index_file)
        else:
            self.indexer = CBIRIndexer()  # Crear un indexador por defecto
    
    def load_index(self, index_file):
        """
        Carga un índice desde un archivo.
        
        Args:
            index_file: Ruta del archivo de índice
            
        Returns:
            True si la carga fue exitosa, False en caso contrario
        """
        try:
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
                
            self.image_features = index_data['features']
            self.hog_images = index_data['hog_images']
            
            # Crear el indexador con los mismos parámetros
            params = index_data['params']
            self.indexer = CBIRIndexer(
                feature_bins=params['feature_bins'],
                hog_cell_size=params['hog_cell_size'],
                hog_block_size=params['hog_block_size']
            )
            
            print(f"Índice cargado correctamente con {len(self.image_features)} imágenes")
            return True
        except Exception as e:
            print(f"Error al cargar el índice: {e}")
            self.indexer = CBIRIndexer()  # Crear un indexador por defecto en caso de error
            return False

    def select_query_image(self, image_folder=None):
        """
         Abre un exlorador de arhivos para seleccionar la imagen de consulta.
        
         Args:
             image_folder: Directorio inicial para el diálogo
            
         Returns:
             Ruta de la imagen seleccionada o None si se cancela
         """
        root = tk.Tk()
        root.withdraw()  # Ocultar la ventana principal

        # Abrir un cuadro de diálogo para seleccionar la imagen
        archivo = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png")],
            initialdir=image_folder if image_folder else "/"
        )
        return archivo
        
    def search(self, query_image_path, top_n=5):
        """
        Busca imágenes similares a la imagen de consulta.
        
        Args:
            query_image_path: Ruta de la imagen de consulta
            top_n: Número de imágenes similares a retornar
            
        Returns:
            Tuple con lista de tuplas (nombre_imagen, distancia) y la imagen HOG de consulta
            o ([], None) en caso de error
        """
        if not self.indexer:
            print("Error: El indexador no está disponible")
            return [], None
            
        if not os.path.exists(query_image_path):
            print(f"Error: La imagen de consulta no existe en {query_image_path}")
            return [], None
            
        # Extraer características de la imagen de consulta
        query_color_hist, query_hog_features, query_hog_image = self.indexer.extract_features(query_image_path)
        
        if query_color_hist is None or query_hog_features is None:
            print("Error: No se pudieron extraer características de la imagen de consulta")
            return [], None
            
        query_features = np.hstack([query_color_hist, query_hog_features])
        query_image_name = os.path.basename(query_image_path)
        
        # Verificar si hay imágenes indexadas
        if not self.image_features:
            print("Advertencia: No hay imágenes indexadas para comparar")
            return [], query_hog_image
        
        # Calcular distancias (usamos distancia euclidiana)
        similarities = {}
        for name, features in self.image_features.items():
            try:
                similarities[name] = distance.euclidean(query_features, features)
            except Exception as e:
                print(f"Error al calcular distancia para {name}: {e}")
        
        # Eliminar la imagen de consulta de las distancias si está en el índice
        if query_image_name in similarities:
            del similarities[query_image_name]
        
        # Ordenar por similitud y tomar los top_n
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])[:top_n]
        
        return sorted_similarities, query_hog_image
    
    def plot_distance_graph(self, similar_images, query_image_name):
        """
        Grafica las distancias como puntos en un scatter plot.
        
        Args:
            similar_images: Lista de tuplas (nombre_imagen, distancia)
            query_image_name: Nombre de la imagen de consulta
        """
        if not similar_images:
            print("No hay distancias para graficar")
            return
            
        names = [name for name, _ in similar_images]
        distances = [dist for _, dist in similar_images]
        
        plt.figure(figsize=(12, 6))
        plt.scatter(range(len(distances)), distances, c='blue', label="Imágenes similares")
        plt.axhline(y=0, color='r', linestyle='--', label="Imagen de consulta")
        plt.xticks(range(len(distances)), names, rotation=90)
        plt.xlabel("Imagen")
        plt.ylabel("Distancia Euclidiana")
        plt.title(f"Distancia de cada imagen a la imagen de consulta ({query_image_name})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def display_similar_images(self, query_image_path, similar_images, image_folder):
        """
        Muestra las imágenes más similares a la consulta con sus distancias.
        
        Args:
            query_image_path: Ruta de la imagen de consulta
            similar_images: Lista de tuplas (nombre_imagen, distancia)
            image_folder: Carpeta donde se encuentran las imágenes
        """
        # Verificar si hay imágenes para mostrar
        if not os.path.exists(query_image_path):
            print(f"La imagen de consulta no existe: {query_image_path}")
            return
            
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            print("No se pudo cargar la imagen de consulta")
            return
            
        if not similar_images:
            # Mostrar solo la imagen de consulta
            plt.figure(figsize=(8, 8))
            plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
            plt.title("Imagen de consulta (sin resultados similares)")
            plt.axis('off')
            plt.show()
            return
            
        # Mostrar la imagen de consulta y las imágenes similares
        num_images = len(similar_images) + 1  # +1 para la imagen de consulta
        fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
        
        # Si solo hay una imagen (solo consulta), convertir axes a array
        if num_images == 1:
            axes = np.array([axes])
        
        # Mostrar imagen de consulta
        axes[0].imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Consulta")
        axes[0].axis('off')
        
        # Mostrar imágenes similares
        for i, (name, dist) in enumerate(similar_images):
            image_path = os.path.join(image_folder, name)
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    axes[i+1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    axes[i+1].set_title(f"#{i+1}: Dist: {dist:.2f}")
                    axes[i+1].axis('off')
                else:
                    axes[i+1].text(0.5, 0.5, f"Error loading\n{name}", 
                                  ha='center', va='center')
                    axes[i+1].axis('off')
            else:
                axes[i+1].text(0.5, 0.5, f"File not found\n{name}", 
                              ha='center', va='center')
                axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()