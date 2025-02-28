#----PACOTES NECESSÁRIOS----#
opencv-python
numpy
tensorflow
nltk
scikit-learn
joblib
flask
-> O ARQUIVO 'instalador de pacotes.bat' PODE INSTALAR TODAS AUTOMÁTICAMENTE


Exercício: Desenvolvimento de uma Aplicação de Machine Learning para Identificação de um Bicho-Preguiça via Imagem ou Texto

📖 Descrição da Tarefa
Neste exercício, você desenvolverá um modelo de Machine Learning capaz de identificar se um dado representa um bicho-preguiça ou não. Para isso, a aplicação deverá receber uma imagem ou um texto e classificar corretamente o conteúdo.

Se a entrada for uma imagem: O modelo deverá analisar características visuais e classificar se o animal na foto é um bicho-preguiça ou não.
Se a entrada for um texto: O modelo deverá interpretar a descrição fornecida e determinar se o texto faz referência a um bicho-preguiça ou a outro animal.
Para isso, será necessário desenvolver dois modelos de Machine Learning:

Modelo de Visão Computacional (CNN) para processar imagens e reconhecer padrões característicos do bicho-preguiça.
Modelo de Processamento de Linguagem Natural (NLP) para analisar descrições textuais e identificar menções ao bicho-preguiça.

🎯 Objetivos
Criar e organizar um conjunto de dados com imagens de bichos-preguiça e outros animais.
Pré-processar as imagens redimensionando e normalizando os dados para o treinamento da CNN.
Criar um modelo de CNN capaz de classificar corretamente imagens de bichos-preguiça.
Criar um conjunto de textos contendo descrições de bichos-preguiça e de outros animais.
Treinar um modelo de NLP para classificar textos corretamente.
Implementar uma aplicação que aceite entrada de texto ou imagem e retorne o resultado da classificação.

🔍 Instruções
Utilize Python e bibliotecas como TensorFlow/Keras, OpenCV, NumPy, NLTK e Flask.
Para imagens, treine uma Rede Neural Convolucional (CNN) para classificação.
Para textos, utilize TF-IDF e um modelo de classificação (SVM, Random Forest ou Naive Bayes).
Permita que o usuário faça upload de uma imagem ou insira um texto e receba a classificação como resposta.
Caso opte por desenvolver uma API, utilize Flask ou FastAPI para criar um endpoint que receba a entrada e retorne a classificação.

✅ Critérios de Avaliação
Implementação Correta	O código aceita entrada de imagem e texto e classifica corretamente?	30%
Organização dos Dados	O conjunto de dados de imagens e textos está bem estruturado?	20%
Precisão dos Modelos	Os modelos atingem uma taxa de acurácia aceitável?	20%
Documentação e Clareza	O código está bem comentado e explicado?	15%
Testes e Interpretação	O modelo foi testado corretamente com novas imagens e textos?	15%

💡 Desafio Extra (Bônus)
Implementar Transfer Learning utilizando um modelo pré-treinado (ResNet, MobileNet).
Criar uma interface web interativa para upload de imagens e inserção de textos.
Implementar detecção em tempo real via webcam utilizando OpenCV.