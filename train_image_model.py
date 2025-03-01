"""
import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model #type: ignore (meu VSCode detecta um erro falso aqui. Perdi um dia inteiro tentando resolver essa alucinação dele...)
import joblib  # Para carregar modelos e vetores
from image_processing import load_and_preprocess_image
from text_processing import preprocess_text, vectorize_text
import numpy as np
from PIL import Image  # Adicionado para lidar com erros de leitura de imagem

app = Flask(__name__)

# Carrega os modelos treinados
image_classifier = load_model('models/image_classifier.h5')
text_classifier = joblib.load('models/text_classifier.pkl')
# Extrai o vetorizador TF-IDF do pipeline
tfidf_vectorizer = text_classifier.named_steps['tfidf']


@app.route('/', methods=['GET'])
def index():
    return "<h1>Sloth Identifier API</h1><p>Use /predict to classify images or text.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        # Processa imagem
        image_file = request.files['image']
        image_path = "temp_image.jpg"  # Salva temporariamente
        image_file.save(image_path)

        try:
            processed_image = load_and_preprocess_image(image_path)
            if processed_image is None:
                os.remove(image_path)
                return jsonify({'error': 'Failed to process image'}), 400

            processed_image = np.expand_dims(processed_image, axis=0)  # Adiciona dimensão do lote

            prediction = image_classifier.predict(processed_image)[0][0]  # Obtém a probabilidade
            result = "Sloth" if prediction > 0.5 else "Not Sloth"
            os.remove(image_path)
            return jsonify({'result': result, 'probability': float(prediction)})

        except Exception as e:
            os.remove(image_path)  # Garante que o arquivo temporário seja removido
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500


    elif 'text' in request.form:
        # Processa texto
        text = request.form['text']

        # Vetoriza o texto usando o vetorizador TF-IDF carregado
        vectorized_text = tfidf_vectorizer.transform([text])

        prediction = text_classifier.predict_proba(vectorized_text)[0][1] # Corrected index to [0][1] to access the probability of class 'Sloth'
        result = "Sloth" if prediction > 0.5 else "Not Sloth"

        return jsonify({'result': result, 'probability': float(prediction)})

    else:
        return jsonify({'error': 'No image or text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model #type: ignore (meu VSCode detecta um erro falso aqui. Perdi um dia inteiro tentando resolver essa alucinação dele...)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Resizing, Rescaling #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from image_processing import load_and_preprocess_image, create_data_augmentation_layers

# Defina os caminhos dos diretórios de dados
data_dir = 'data/images/'
sloth_dir = os.path.join(data_dir, 'sloth')
not_sloth_dir = os.path.join(data_dir, 'not_sloth')

# Defina o tamanho desejado para as imagens
img_width, img_height = 224, 224

# Caminho do modelo
model_path = 'models/image_classifier.h5'

# Função para carregar e pré-processar as imagens
def load_data(data_dir, img_width, img_height):
    images = []
    labels = []

    for category in ['sloth', 'not_sloth']:
        path = os.path.join(data_dir, category)
        class_num = 0 if category == 'sloth' else 1
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = load_and_preprocess_image(img_path, target_size=(img_width, img_height))
                if image is not None:
                    images.append(image)
                    labels.append(class_num)
            except Exception as e:
                print(f"Erro ao carregar a imagem {img}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Carrega os dados
images, labels = load_data(data_dir, img_width, img_height)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Cria as camadas de data augmentation
data_augmentation = create_data_augmentation_layers()

# Verifica se o modelo existe
if os.path.exists(model_path):
    # Carrega o modelo existente
    print(f"Carregando modelo existente de: {model_path}")
    model = load_model(model_path)
else:
    # Cria um novo modelo
    print("Criando um novo modelo...")
    model = Sequential([
        # Camadas de Resizing e Rescaling para garantir tamanho e normalização consistentes
        Resizing(img_width, img_height),
        Rescaling(1./255),  # Normaliza os valores dos pixels para [0, 1]
        data_augmentation, # Aplica data augmentation

        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Camada de saída com sigmoid para classificação binária
    ])

    # Compila o modelo
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Treina o modelo
batch_size = 32
epochs = 20

print("Iniciando o treinamento...")
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test, y_test))

# Salva o modelo
model.save(model_path)

print(f"Modelo de imagem treinado e salvo em: {model_path}")