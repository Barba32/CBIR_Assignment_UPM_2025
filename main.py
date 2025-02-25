import os
from utils import CBIRIndexer, CBIRSearcher

def main():
    """Función principal para ejecutar el sistema CBIR."""
    image_folder = "./holidays_database"  # Carpeta con las imágenes a indexar
    index_file = "./cbir_index.pkl"  # Archivo donde se guardará el índice
    image_query = "./holidays_test" # Carpeta con las imágenes de prueba
    
    # Preguntar si se quiere crear un nuevo índice o usar uno existente
    create_new_index = input("¿Desea crear un nuevo índice? (s/n): ").lower() == 's'
    
    if create_new_index:
        indexer = CBIRIndexer()
        if indexer.index_folder(image_folder):
            indexer.save_index(index_file)
    
    # Inicializar el buscador
    searcher = CBIRSearcher(index_file)
    
    # Buscar con imagen seleccionada por el usuario
    query_image_path = searcher.select_query_image(image_query)
    
    if query_image_path:
        print(f"Buscando imágenes similares a {os.path.basename(query_image_path)}...")
        similar_images, query_hog_image = searcher.search(query_image_path)
        
        # Obtener el nombre de la imagen de consulta
        query_image_name = os.path.basename(query_image_path)
        
        # Mostrar resultados
        if similar_images:
            print("Las imágenes más similares son:")
            for i, (name, dist) in enumerate(similar_images):
                print(f"{i+1}. {name} (Distancia: {dist:.4f})")
        else:
            print("No se encontraron imágenes similares")
            
        # Mostrar visualizaciones
        searcher.display_similar_images(query_image_path, similar_images, image_folder)
        if similar_images:
            searcher.plot_distance_graph(similar_images, query_image_name)
    else:
        print("No se seleccionó ninguna imagen")


if __name__ == "__main__":
    main()