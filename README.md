# Transfer Learning: Classificação de Gatos vs. Cachorros
<hr>
Este repositório contém um projeto desenvolvido em Python (utilizando o VSCode) que demonstra a aplicação de **Transfer Learning** para classificação de imagens. O projeto utiliza o modelo pré-treinado **MobileNetV2** do TensorFlow Hub como extrator de características, adicionando uma camada densa final para classificar imagens em duas categorias: gatos e cachorros.

## Visão Geral

- **Objetivo:**  
  Aproveitar o aprendizado prévio de uma rede neural (MobileNetV2) para classificar imagens de gatos e cachorros, mesmo com um conjunto de dados reduzido.

- **Dataset:**  
  Imagens redimensionadas para 224x224 pixels. O projeto utiliza imagens dos diretórios:
  - `./kagglecatsanddogs_5340/DogResized/`
  - `./kagglecatsanddogs_5340/CatResized/`  
  Você pode substituir essas imagens por sua própria base de dados (ex.: fotos pessoais, de amigos, etc).

## Estrutura do Projeto

- **main.py:**  
  Script principal contendo:
  - Carregamento e pré-processamento das imagens.
  - Divisão do dataset em conjuntos de treino e teste.
  - Construção, compilação e treinamento da rede neural.
  - Sistema de predição que solicita um caminho de imagem para classificação.

- **Pastas de imagens:**  
  - `DogResized/`: Contém as imagens de cachorros.  
  - `CatResized/`: Contém as imagens de gatos.

## Requisitos

- **Linguagem:** Python 3.x
- **Bibliotecas:** 
  - numpy
  - Pillow (PIL)
  - matplotlib
  - scikit-learn
  - OpenCV
  - TensorFlow
  - TensorFlow Hub

## Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/PixoPjl/Transfer-Learning-py/

2. **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
