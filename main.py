import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import cv2
import os

# Path para imagens de cachorros e gatos redimensionadas para 224x224.
dog_resized_folder = './kagglecatsanddogs_5340/DogResized/'
cat_resized_folder = './kagglecatsanddogs_5340/CatResized/'

labels = []
cats_and_dogs = []

# Lista os nomes dos arquivos com extensão .jpg na pasta de cachorros
dog_filenames = [f for f in os.listdir(dog_resized_folder) if f.endswith('.jpg')]

# Separa o nome e a extensão do arquivo e ajusta em ordem númerica com sorted.
dog_filenames = sorted(dog_filenames, key=lambda x: int(x.split('.')[0]))

# Itera sobre cada imagem de cachorro
for filename in dog_filenames:
    img_path = os.path.join(dog_resized_folder, filename) # Monta o caminho completo
    try:
        img = Image.open(img_path)          # Abre a imagem
        img_array = np.array(img)           # Converte a imagem em array NumPy
        cats_and_dogs.append(img_array)     # Adiciona o array à lista cats_and_dogs
        labels.append(1)                    # Adiciona o label 1 (cachorro)
    except Exception as e:
        print(f'Erro ao processar {img_path}: {e}')

cat_filenames = [f for f in os.listdir(cat_resized_folder) if f.endswith('.jpg')]
cat_filenames = sorted(cat_filenames, key=lambda x: int(x.split('.')[0]))

# Itera sobre cada imagem de gato
for filename in cat_filenames:
    img_path = os.path.join(cat_resized_folder, filename)
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        cats_and_dogs.append(img_array)
        labels.append(0)                    # Label 0 para gato
    except Exception as e:
        print(f'Erro ao processar {img_path}: {e}')

dog_cat_images = np.array(cats_and_dogs)    # Cria um array com numpy de cats_and_dogs
labels = np.array(labels)                   # Cria um array com numpy de labels

X = dog_cat_images
Y = np.asarray(labels)

# ---- Divisão de Treino e Teste

# Test_Size definido para 0.2, logo temos: 200 Imagens de Teste e 800 Imagens de Treino
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Normalização dos dados dividindo por 255 para atingir o estilo '0' e '1' de normalização.
X_train_scaled = X_train/255
X_test_scaled = X_test/255

# X_train_scaled: Conjunto de dados previamente escalado/divido por 255 para normalizar os pixels em valores de '0' e '1'.
# Y_train: Conjunto de rótulos correspondentes às imagens de treinamento, indicando se cada imagem representa um gato(0) ou cachorro(1).


# ---- Construindo a Rede Neural

import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
from tf_keras.layers import Dense

# Carrega o modelo Pre-Treinado MobileNetV2 a partir do TensorFlow HUB
mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

# Trainable=False para não ajustar os pesos e preservar o conhecimento anterior | 224 de altura, 224 de largura, 3 canais de cores: RGB
pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3), trainable=False)

# Definindo número de classes, Gato e Cachorro: 2
num_of_classes = 2

model = tf_keras.Sequential([
    pretrained_model, #
    tf_keras.layers.Dense(num_of_classes) # Define 2 neurônios
])

model.compile(
    optimizer = 'adam',
    loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics  = ['acc']
)

# Percorre todo o conjunto de dados de treinamento 5 vezes, treinando por 5 epocas.
model.fit(X_train_scaled, Y_train, epochs=5)

score, acc = model.evaluate(X_test_scaled, Y_test)

# ---- Sistema preditivo

# Recebe o caminho para imagem.
input_image_path = input('Caminho da imagem para ser processada: ')

# Lê a imagem
input_image = cv2.imread(input_image_path)

# Exibe a imagem
cv2.imshow('', input_image)

# Redimensiona a imagem para 224x224 conforme o esperado pelo modelo.
input_image_resize = cv2.resize(input_image, (224, 224))

# Normalização dos pixels para '0' e '1'.
input_image_scaled = input_image_resize/255

# Remodelação para incluir a dimensão do batch.
image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

# Prevê a classe da imagem
input_prediction = model.predict(image_reshaped)

# argmax retorna o índice da classe com maior probabilidade
input_pred_label = np.argmax(input_prediction)

# Se for '0' é gato, caso contrario é cachorro.
if input_pred_label == 0:
    print('A imagem representa um Gato')

else:
    print('A imagem representa um Cachorro')