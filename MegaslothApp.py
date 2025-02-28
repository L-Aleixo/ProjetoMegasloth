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