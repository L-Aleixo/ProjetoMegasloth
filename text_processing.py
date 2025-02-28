import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def download_nltk_resources(): #Checa e baixa os recursos da NLTK
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()

def preprocess_text(text): #Pré-processa o texto
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)  #Remove pontuação
    text = text.lower()  #Converte para minúsculas
    stop_words = stopwords.words('portuguese')  #Obtém stopwords
    text = [word for word in text.split() if word not in stop_words]  #Remove stopwords
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]  #Lematiza
    text = " ".join(text)  #Junta as palavras de volta em um texto
    return text


def create_tfidf_vectorizer(texts): #Cria e treina um vetorizador TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)  #Treina o vetorizador com os textos
    return vectorizer


def vectorize_text(text, vectorizer): #Vetoriza um texto usando um vetorizador TF-IDF treinado
    text = preprocess_text(text)  #Pré-processa o texto
    vectorized_text = vectorizer.transform([text])  #Vetoriza o texto
    return vectorized_text


if __name__ == '__main__':
    #Teste das funções de pré-processamento
    example_text = "Matar dois coelhos com uma caixa d'água só!"
    preprocessed_text = preprocess_text(example_text)
    print(f"\nTexto pré-processado: {preprocessed_text}")

    #Criação e teste do vetorizador TF-IDF
    texts = ["Este é o primeiro documento.",
        "Este documento é o segundo documento.",
        "E este é o terceiro.",
        "Quarto documento que não é como os outros."]
    vectorizer = create_tfidf_vectorizer(texts)
    vectorized_text = vectorize_text(example_text, vectorizer)
    print(f"Formato do texto vetorizado: {vectorized_text.shape}")