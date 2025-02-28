import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers #type: ignore (meu VSCode detecta um erro falso aqui. Perdi um dia inteiro tentando resolver essa alucinação dele...)

def load_and_preprocess_image(image_path, target_size=(224,224)): #Carrega, redimensiona e normaliza uma imagem
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Imagem não encontrada ou não pode ser carregada.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converte para RGB
        img = cv2.resize(img, target_size)
        img = img.astype("float32") #Força o formato float32
        img = img / 255.0 #Normaliza
        return img
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None

def create_data_augmentation_layers(rotation_range=0.2, zoom_range=0.2, flip_mode="horizontal"): #Cria camadas de data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(rotation_range),
        layers.RandomZoom(zoom_range),
        layers.RandomFlip(flip_mode)
    ])
    return data_augmentation

if __name__ == '__main__':
    example_image_path = "data/images/sloth/image2.jpg" #Caminho de imagem provisório
    processed_image = load_and_preprocess_image(example_image_path)

    if processed_image is not None:
        #Teste da função de pré-processamento
        print(f"Formato da imagem processada: {processed_image.shape}")
        print(f"Valor máximo da imagem processada: {np.max(processed_image)}")

        #Teste das camadas de data augmentation
        data_augmentation_layers = create_data_augmentation_layers()
        augmented_image = data_augmentation_layers(tf.expand_dims(processed_image, axis=0)) #Dimensão do lote

        print(f"Formato da imagem aumentada: {augmented_image.shape}")
    else: print("Falha ao processar a imagem de exemplo.")