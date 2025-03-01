import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib  # Para salvar modelos e vetores
from text_processing import preprocess_text

# Defina os caminhos dos diretórios de dados
data_dir = 'data/texts/'
sloth_dir = os.path.join(data_dir, 'sloth')
not_sloth_dir = os.path.join(data_dir, 'not_sloth')

# Função para carregar e pré-processar os textos
def load_data(data_dir):
    texts = []
    labels = []

    for category in ['sloth', 'not_sloth']:
        path = os.path.join(data_dir, category)
        class_num = 0 if category == 'sloth' else 1
        for file in os.listdir(path):
            try:
                file_path = os.path.join(path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    text = preprocess_text(text)  # Pré-processa o texto
                    texts.append(text)
                    labels.append(class_num)
            except Exception as e:
                print(f"Erro ao carregar o texto {file}: {e}")

    texts = np.array(texts)
    labels = np.array(labels)
    return texts, labels

# Carrega os dados
texts, labels = load_data(data_dir)

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Cria um pipeline com TF-IDF e um classificador SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear', probability=True))  # Kernel linear geralmente funciona bem para texto
])

# Treina o pipeline
pipeline.fit(X_train, y_train)

# Faz previsões nos dados de teste
y_pred = pipeline.predict(X_test)

# Calcula a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy}")

# Salva o modelo treinado
model_filename = 'models/text_classifier.pkl'
joblib.dump(pipeline, model_filename)
print(f"Modelo de texto treinado e salvo em {model_filename}")